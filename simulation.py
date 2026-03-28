import joblib
import pandas as pd
import numpy as np
import os

FACTORIES = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory",
]

# Geographic lead-time multipliers: factory → region → multiplier
FACTORY_REGION_MULT = {
    "Lot's O' Nuts":     {"Pacific": 0.85, "Atlantic": 1.20, "Interior": 1.00, "Gulf": 1.05},
    "Wicked Choccy's":   {"Pacific": 1.25, "Atlantic": 0.80, "Interior": 1.10, "Gulf": 0.95},
    "Sugar Shack":       {"Pacific": 1.10, "Atlantic": 1.15, "Interior": 0.85, "Gulf": 1.00},
    "Secret Factory":    {"Pacific": 1.15, "Atlantic": 1.05, "Interior": 0.90, "Gulf": 1.05},
    "The Other Factory": {"Pacific": 1.20, "Atlantic": 1.00, "Interior": 0.95, "Gulf": 0.90},
}


def _avg_mult(factory: str) -> float:
    return sum(FACTORY_REGION_MULT[factory].values()) / 4


def simulate_reallocation(base_lead: float, base_profit: float,
                           current_factory: str) -> list[dict]:
    """
    Simulate assigning a product to each factory.
    Returns a list of dicts sorted by predicted lead time (ascending).
    """
    results = []
    for factory in FACTORIES:
        pred_lead   = round(base_lead   * _avg_mult(factory), 2)
        # Profit sensitivity: 0.8 % degradation per extra day
        profit_adj  = round(base_profit * (1 - (pred_lead - base_lead) * 0.008), 2)
        lead_change = round(pred_lead - base_lead, 2)
        results.append({
            "Factory":         factory,
            "Predicted Lead":  pred_lead,
            "Lead Change":     lead_change,
            "Profit Impact":   profit_adj,
            "Is Current":      factory == current_factory,
        })

    results.sort(key=lambda x: x["Predicted Lead"])
    return results


def bulk_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every unique product, find the factory that minimises predicted lead time.
    Returns a DataFrame of reallocation recommendations.
    """
    from preprocessing import FACTORY_MAP

    records = []
    for product, grp in df.groupby("Product Name"):
        current_factory = FACTORY_MAP.get(product, "Unknown")
        base_lead       = grp["Lead_Time"].mean()
        base_profit     = grp["Gross Profit"].sum()

        sims = simulate_reallocation(base_lead, base_profit, current_factory)
        best = sims[0]

        records.append({
            "Product":          product,
            "Division":         grp["Division"].iloc[0],
            "Current Factory":  current_factory,
            "Best Factory":     best["Factory"],
            "Current Lead":     round(base_lead, 2),
            "Best Lead":        best["Predicted Lead"],
            "Lead Saving":      round(base_lead - best["Predicted Lead"], 2),
            "Profit Impact":    best["Profit Impact"],
            "Reassign":         best["Factory"] != current_factory,
        })

    return (
        pd.DataFrame(records)
        .sort_values("Lead Saving", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from preprocessing import load_data
    df = load_data()
    recs = bulk_recommendations(df)
    print(recs.to_string(index=False))
