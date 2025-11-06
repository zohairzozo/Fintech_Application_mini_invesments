import streamlit as st
from equity import _recompute_equity_value
def _fund_sell(fund_name: str, units: float, price: float, fee: float):
    """
    Sell fund units (amount-based asset).
    Updates cash_balance, fund_holdings, profit_value, txns, equity.
    """
    import pandas as pd

    if units <= 0:
        return None

    pos = st.session_state.fund_holdings.get(fund_name)
    if not pos or float(pos.get("qty", 0.0)) < float(units):
        st.warning("Not enough units to sell.")
        return None

    qty_before = float(pos["qty"])
    avg_cost = float(pos.get("avg_cost", 0.0))

    # P/L on the sold portion
    realized_pl = (float(price) - avg_cost) * float(units) - float(fee)
    gross = float(units) * float(price)
    proceeds = gross - float(fee)

    # Update holding
    new_qty = qty_before - float(units)
    if new_qty <= 1e-9:
        st.session_state.fund_holdings.pop(fund_name, None)
    else:
        # avg_cost stays the same for remaining units
        pos["qty"] = new_qty
        pos["last_price"] = float(price)
        st.session_state.fund_holdings[fund_name] = pos

    # Cash & P/L
    st.session_state.cash_balance = float(st.session_state.cash_balance) + proceeds
    st.session_state.profit_value = float(st.session_state.get("profit_value", 0.0)) + float(realized_pl)

    # Log
    st.session_state.txns.append({
        "ts": pd.Timestamp.utcnow(),
        "side": "SELL",
        "asset_type": "FUND",
        "symbol": fund_name,
        "qty": float(units),
        "price": float(price),
        "fee": float(fee),
        "realized_pl": float(realized_pl),
    })

    _recompute_equity_value()
    return realized_pl, proceeds
