# Tashkent Property Price Command Center

A Pearson BTEC Level 6 applied AI capstone prototype for estimating apartment asking prices from OLX Uzbekistan real-estate listings.

## Project purpose

This project is not a simple price calculator. It is a decision-support command center that:

- audits noisy scraped OLX real-estate data;
- cleans string-based price, area, floor and property attributes;
- trains multiple regression models against a dummy baseline;
- reports MAE, RMSE, R², MAPE and within-tolerance metrics;
- estimates apartment asking prices from user inputs;
- simulates valuation scenarios;
- explains price drivers through permutation importance;
- exports an evidence pack for academic assessment.

## Dataset

Expected dataset file in the repository root:

```text
olx_massive_real_estate.xlsx
```

The app can also accept the dataset through the Streamlit file uploader. This is useful if GitHub upload causes naming or file-format issues.

## Repository structure

```text
app.py
ml_pipeline.py
requirements.txt
README.md
olx_massive_real_estate.xlsx
.streamlit/config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Use Streamlit Community Cloud:

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

## Important limitations

- OLX listings show asking prices, not verified transaction prices.
- Scraped data may contain duplicated, inconsistent or exaggerated fields.
- Predictions should be used as analytical estimates, not final valuation decisions.
- A real commercial deployment would need live data validation, transaction records and monitoring.
