import pandas as pd
import numpy as np

FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":    "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":            "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":       "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":           "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":    "Wicked Choccy's",
    "Laffy Taffy":                          "Sugar Shack",
    "SweeTARTS":                            "Sugar Shack",
    "Nerds":                                "Sugar Shack",
    "Fun Dip":                              "Sugar Shack",
    "Fizzy Lifting Drinks":                 "Sugar Shack",
    "Everlasting Gobstopper":               "Secret Factory",
    "Lickable Wallpaper":                   "Secret Factory",
    "Wonka Gum":                            "Secret Factory",
    "Hair Toffee":                          "The Other Factory",
    "Kazookles":                            "The Other Factory",
}

SHIP_MODE_LEAD = {
    "Same Day":       1.0,
    "First Class":    3.0,
    "Second Class":   5.0,
    "Standard Class": 8.0,
}

REGION_NOISE = {
    "Atlantic": 0.5,
    "Gulf":     0.8,
    "Interior": 1.0,
    "Pacific":  1.5,
}


def load_data(path: str = "data/orders.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  errors="coerce")

    # Derive lead time from ship mode + region (ship dates are placeholder future dates)
    df["Lead_Time"] = (
        df["Ship Mode"].map(SHIP_MODE_LEAD)
        + df["Region"].map(REGION_NOISE).fillna(1.0)
    )

    # Assign factory based on product
    df["Factory"] = df["Product Name"].map(FACTORY_MAP)

    # Feature engineering
    df["Month"]         = df["Order Date"].dt.month
    df["DayOfWeek"]     = df["Order Date"].dt.dayofweek
    df["Quarter"]       = df["Order Date"].dt.quarter
    df["Profit_Margin"] = (df["Gross Profit"] / df["Sales"].replace(0, np.nan)).fillna(0)
    df["Sales_Per_Unit"]= (df["Sales"] / df["Units"].replace(0, np.nan)).fillna(0)

    # Drop rows with missing critical fields
    df = df.dropna(subset=["Lead_Time", "Region", "Ship Mode", "Division", "Product Name"])

    return df


def prepare_features(df: pd.DataFrame):
    feature_cols = ["Month", "DayOfWeek", "Quarter", "Sales", "Units",
                    "Gross Profit", "Cost", "Profit_Margin", "Sales_Per_Unit",
                    "Region", "Ship Mode", "Division", "Product Name", "Factory"]

    df_feat = df[feature_cols + ["Lead_Time"]].copy()

    df_feat = pd.get_dummies(
        df_feat,
        columns=["Region", "Ship Mode", "Division", "Product Name", "Factory"],
        drop_first=False
    )

    X = df_feat.drop("Lead_Time", axis=1)
    y = df_feat["Lead_Time"]

    return X, y
