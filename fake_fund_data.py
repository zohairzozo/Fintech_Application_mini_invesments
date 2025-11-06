import pandas as pd
import numpy as np
import datetime as dt

def _fake_fund_data():
    """Simulate time-series for 5 investment funds."""
    np.random.seed(42)
    base_date = pd.date_range(dt.date.today() - dt.timedelta(days=180), periods=180)
    
    funds = {
        "OP-Rohkea A": 3,
        "OP-Kehittyvät markkinat A": 4,
        "OP-Suomi Indeksi A": 5,
        "OP-Kiina A": 2,
        "OP-Maailma A": 4
    }

    data = {}
    for fund, stars in funds.items():
        base_price = 100 + np.random.normal(0, 2)
        noise = np.random.normal(0, 1, len(base_date)).cumsum()
        prices = base_price + noise
        df = pd.DataFrame({
            "Date": base_date,
            "Price": prices.round(2)
        })
        df["Volatility"] = np.random.uniform(5, 20)  # %
        df["Expenses"] = np.random.uniform(0.5, 2)   # %
        df["Morning_Star"] = stars
        data[fund] = df
    return data

