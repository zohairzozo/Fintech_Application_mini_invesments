# funds_real_data.py

import pandas as pd
from functools import lru_cache
from pathlib import Path

# Adjust if your path/layout is different
EXCEL_PATH = Path(__file__).with_name("Investment_Funds_Data.xlsx")

# Hardcoded metadata used in Subpage 2 UI
MORNINGSTAR_STARS = {
    "OP-Korkosalkku A": 3,
    "OP-Kiina A": 4,
    "OP-Rohkea A": 5,
    "OP-Kehittyvät Osakemarkkinat A": 4,
    "OP-Maailma Indeksi A": 3,
    "OP-Suomi Indeksi A": 4,
}

OPERATING_EXPENSES = {  # % per year, example values
    "OP-Korkosalkku A": 0.65,
    "OP-Kiina A": 1.60,
    "OP-Rohkea A": 1.40,
    "OP-Kehittyvät Osakemarkkinat A": 1.70,
    "OP-Maailma Indeksi A": 0.30,
    "OP-Suomi Indeksi A": 0.20,
}


def _normalize(name: str) -> str:
    # Make Excel names & display names match
    return (name or "").replace("(kasvu)", "").strip()


@lru_cache()
def load_fund_data() -> dict:
    """
    Returns:
        {
          "OP-Kiina A": {
              "name": ...,
              "nav": DataFrame[date, navPerShare],
              "latest_nav": float,
              "volatility": float | None,
              "growth": float | None,
              "growth_from_date": pd.Timestamp | None,
              "morning_star": int | None,
              "operating_expense": float | None,
          },
          ...
        }
    """
    xls = pd.ExcelFile(EXCEL_PATH)

    # --- Growth ---
    growth = pd.read_excel(xls, "Growth of Funds")
    
    growth["FundKey"] = growth["secname"].map(_normalize)

    # --- Volatility ---
    vol = pd.read_excel(xls, "Volatility")
    vol = vol.rename(columns={"Fund Name": "FundKey"})

    data = {}

    for sheet in xls.sheet_names:
        if sheet in ("Growth of Funds", "Volatility"):
            continue

        # NAV timeseries
        nav = pd.read_excel(xls, sheet)
        nav["Date"] = pd.to_datetime(nav["Date"])
        nav = nav.sort_values("Date").dropna(subset=["Date", "navPerShare"])

        display_name = _normalize(str(sheet))

        # Match metadata rows
        g_row = growth[growth["FundKey"] == display_name]
        v_row = vol[vol["FundKey"] == display_name]

        data[display_name] = {
            "name": display_name,
            "nav": nav,
            "latest_nav": float(nav["navPerShare"].iloc[-1]),
            "growth": float(g_row["Growth"].iloc[0]) if not g_row.empty else None,
            "growth_from_date": pd.to_datetime(g_row["FROM_DATE"].iloc[0])
            if not g_row.empty
            else None,
            "volatility": float(v_row["Volatility"].iloc[0]) if not v_row.empty else None,
            "morning_star": MORNINGSTAR_STARS.get(display_name),
            "operating_expense": OPERATING_EXPENSES.get(display_name),
        }

    return data
