import yfinance as yf
import streamlit as st
from fake_fund_data import _fake_fund_data as fund_data
def _recompute_equity_value():
    """
    Recalculate TOTAL equity from current holdings (stocks + funds) using recent prices.
    - Stocks: yfinance (fallback to last_price)
    - Funds: fake_fund_data() latest (fallback to last_price)
    """
    import pandas as pd

    total = 0.0

    # --- Stocks ---
    for sym, pos in st.session_state.get("holdings", {}).items():
        try:
            px = yf.download(sym, period="5d", auto_adjust=False)["Adj Close"].dropna().iloc[-1]
            px = float(px)
        except Exception:
            px = float(pos.get("last_price", 0.0))
        total += float(pos.get("qty", 0.0)) * px
        pos["last_price"] = px  # cache latest

    # --- Funds ---
    # Try to refresh fund prices from your fabricated data function if available
    fund_price_map = {}
    try:
        # You said you used: from fake_fund_data import _fake_fund_data as fund_data
        FUNDS_DATA = fund_data()  # dict: {fund_name: DataFrame with "Price"}
        for fname, df in FUNDS_DATA.items():
            try:
                fund_price_map[fname] = float(df["Price"].iloc[-1])
            except Exception:
                pass
    except Exception:
        pass  # fall back to last_price in holdings if import/lookup fails

    for fname, fpos in st.session_state.get("fund_holdings", {}).items():
        if fname in fund_price_map:
            fpx = fund_price_map[fname]
        else:
            fpx = float(fpos.get("last_price", 0.0))
        total += float(fpos.get("qty", 0.0)) * fpx
        fpos["last_price"] = fpx  # cache latest

    st.session_state.equity_value = round(float(total), 2)
