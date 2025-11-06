import streamlit as st
from equity import _recompute_equity_value  
def _fund_buy(fund_name: str, invest_amount: float, price: float, fee: float, when=None):
    """
    Buy fund units by INVESTED AMOUNT (not shares).
    - invest_amount: total cash to spend (includes fee)
    - price: current fund NAV/price
    - fee: fixed fee per order
    """
    import pandas as pd

    if invest_amount <= 0:
        st.warning("Enter a positive amount to invest.")
        return None
    if st.session_state.cash_balance < invest_amount:
        st.warning("Not enough cash balance.")
        return None
    if invest_amount <= fee:
        st.warning("Amount must be greater than fee.")
        return None

    units = (float(invest_amount) - float(fee)) / float(price)
    if units <= 0:
        st.warning("Computed units is zero — increase the amount.")
        return None

    # Update cash
    st.session_state.cash_balance = float(st.session_state.cash_balance) - float(invest_amount)

    # Update holding (weighted avg)
    pos = st.session_state.fund_holdings.get(fund_name, {"qty": 0.0, "avg_cost": 0.0, "last_price": price})
    old_qty = float(pos.get("qty", 0.0))
    old_cost = float(pos.get("avg_cost", 0.0))
    new_qty = old_qty + units
    new_cost = ((old_qty * old_cost) + (units * price)) / new_qty if new_qty > 0 else 0.0

    pos.update({
        "qty": new_qty,
        "avg_cost": new_cost,
        "last_price": float(price),
    })
    st.session_state.fund_holdings[fund_name] = pos

    # Log txn (reuse your txns array)
    st.session_state.txns.append({
        "ts": (pd.Timestamp.utcnow() if when is None else pd.to_datetime(when)),
        "side": "BUY",
        "asset_type": "FUND",
        "symbol": fund_name,
        "qty": float(units),
        "price": float(price),
        "fee": float(fee),
        "amount": float(invest_amount),
    })

    # If you later compute a combined equity (stocks + funds), call it here
    _recompute_equity_value()

    return units
