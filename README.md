# Nassau Candy — Factory Shipping Optimization

Intelligent decision-support system for factory reallocation and shipping optimization.

## Project Structure

```
nassau_candy_project/
├── app.py                 # Streamlit dashboard (main entry point)
├── preprocessing.py       # Data loading & feature engineering
├── simulation.py          # Factory reallocation simulation engine
├── train_model.py         # Model training (Linear / RF / GBM)
├── requirements.txt       # Python dependencies
├── data/
│   └── orders.xlsx        # Source order data
├── models/
│   ├── shipping_model.pkl          # Trained best model
│   └── feature_columns.pkl         # Feature column list
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain model
python train_model.py

# 3. Launch dashboard
streamlit run app.py
```

## Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| Factory Optimization Simulator | Compare predicted lead times across all 5 factories per product |
| What-If Scenario Analysis | Model the effect of priority slider on lead time & profit |
| Recommendation Dashboard | Ranked factory reassignment recommendations |
| Risk & Impact Panel | At-risk orders, profit scatter, regional breakdown |

## Factories

| Factory | Location |
|---------|----------|
| Lot's O' Nuts | Arizona |
| Wicked Choccy's | Georgia |
| Sugar Shack | North Dakota |
| Secret Factory | Iowa |
| The Other Factory | Tennessee |

## KPIs

- **Lead Time Reduction (%)** — Operational efficiency gain
- **Profit Impact Stability** — Financial safety metric
- **Scenario Confidence Score** — Model reliability (82%)
- **Recommendation Coverage** — % of products with actionable advice
