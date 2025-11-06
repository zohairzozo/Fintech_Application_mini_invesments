import pandas as pd
import streamlit as st
from equity import _recompute_equity_value
def _buy(symbol: str, qty: int, price: float, fee: float):
    if qty <= 0:
        return

    pos = st.session_state.holdings.get(symbol, {"qty": 0.0, "avg_cost": 0.0, "last_price": price, "lots": []})
    old_qty = float(pos["qty"])
    old_cost_total = pos["avg_cost"] * old_qty

    # cash effects
    subtotal = qty * price
    total_cost = subtotal + fee
    if total_cost > st.session_state.cash_balance:
        st.warning("Not enough cash.")
        return

    # update position & lots
    old_qty = float(pos["qty"])
    old_cost_total = pos["avg_cost"] * old_qty
    new_qty = old_qty + qty
    new_cost_total = old_cost_total + subtotal + fee
    new_avg_cost = new_cost_total / new_qty if new_qty > 0 else 0.0

    pos["qty"] = new_qty
    pos["avg_cost"] = new_avg_cost
    pos["last_price"] = price
    pos.setdefault("lots", []).append({"qty": float(qty), "cost": float(price)})
    st.session_state.holdings[symbol] = pos

    # cash & log
    st.session_state.cash_balance -= total_cost
    st.session_state.txns.append({
        "ts": pd.Timestamp.utcnow(),
        "side": "BUY",
        "symbol": symbol,
        "qty": int(qty),
        "price": float(price),
        "fee": float(fee),
    })
    _recompute_equity_value()