import pandas as pd
import streamlit as st
from equity import _recompute_equity_value
def _sell(symbol: str, qty: int, price: float, fee: float):
    """
    Sell using FIFO lots. Robust to missing/empty lots by synthesizing from avg_cost.
    Keeps lots and qty reconciled, applies fee once, updates cash, holdings, realized P/L, equity.
    """
    if qty <= 0:
        return None

    pos = st.session_state.holdings.get(symbol)
    if not pos or float(pos.get("qty", 0.0)) < float(qty):
        st.warning("Not enough shares to sell.")
        return None

    # --- Ensure lots exist and reconcile to qty ---
    EPS = 1e-9
    total_qty = float(pos.get("qty", 0.0))
    lots = list(pos.get("lots", [])) or []

    # If no lots tracked yet, synthesize one lot from avg_cost
    if not lots:
        lots = [{"qty": float(total_qty), "cost": float(pos.get("avg_cost", 0.0))}]

    # Reconcile: make sure sum(lots) == total_qty (within tiny tolerance)
    lot_sum = sum(float(l["qty"]) for l in lots)
    if abs(lot_sum - total_qty) > 1e-6:
        # If lots under-report, pad last lot; if over-report, trim from the tail
        if lot_sum < total_qty:
            if lots:
                lots[-1]["qty"] = float(lots[-1]["qty"]) + (total_qty - lot_sum)
            else:
                lots = [{"qty": total_qty, "cost": float(pos.get("avg_cost", 0.0))}]
        else:
            to_trim = lot_sum - total_qty
            i = len(lots) - 1
            while to_trim > EPS and i >= 0:
                take = min(float(lots[i]["qty"]), to_trim)
                lots[i]["qty"] = float(lots[i]["qty"]) - take
                to_trim -= take
                if lots[i]["qty"] <= EPS:
                    lots.pop(i)
                i -= 1

    # Safety: drop any dust lots
    lots = [ {"qty": float(l["qty"]), "cost": float(l["cost"])} for l in lots if float(l["qty"]) > EPS ]

    # --- FIFO consume ---
    remaining = float(qty)
    realized_pl = 0.0
    new_lots = []

    for lot in lots:
        if remaining <= EPS:
            new_lots.append(lot)
            continue
        take = min(remaining, float(lot["qty"]))
        realized_pl += (float(price) - float(lot["cost"])) * take
        left = float(lot["qty"]) - take
        if left > EPS:
            new_lots.append({"qty": left, "cost": float(lot["cost"])})
        remaining -= take

    # If we couldn't satisfy the sell from lots (shouldn't happen given the earlier check)
    if remaining > EPS:
        st.error("Internal lot mismatch while selling. Please refresh and try again.")
        return None

    # Apply fee once per order
    realized_pl -= float(fee)

    # --- Update position ---
    new_qty = float(pos["qty"]) - float(qty)
    if new_qty <= EPS:
        # Position fully closed
        st.session_state.holdings.pop(symbol, None)
    else:
        # Recompute avg cost from remaining lots
        cost_total = sum(float(l["qty"]) * float(l["cost"]) for l in new_lots)
        pos["qty"] = new_qty
        pos["lots"] = new_lots
        pos["avg_cost"] = (cost_total / new_qty) if new_qty > EPS else 0.0
        pos["last_price"] = float(price)
        st.session_state.holdings[symbol] = pos

    # --- Cash & P/L logs ---
    gross = float(qty) * float(price)
    proceeds = gross - float(fee)
    st.session_state.cash_balance = float(st.session_state.cash_balance) + proceeds
    st.session_state.profit_value = float(st.session_state.get("profit_value", 0.0)) + float(realized_pl)

    st.session_state.txns.append({
        "ts": pd.Timestamp.utcnow(),
        "side": "SELL",
        "symbol": symbol,
        "qty": int(qty),
        "price": float(price),
        "fee": float(fee),
        "realized_pl": float(realized_pl),
    })

    _recompute_equity_value()
    return realized_pl, proceeds