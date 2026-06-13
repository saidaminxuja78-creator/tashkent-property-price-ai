from __future__ import annotations

from pathlib import Path
import json
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import (
    TARGET, PREFERRED_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    find_dataset_path, load_raw_data, clean_real_estate_data, market_summary, district_benchmark,
    evaluate_models, split_xy, predict_single, prepare_prediction_features, model_feature_importance,
    bootstrap_metric_ci, learning_curve_table, evidence_pack, save_model, dataset_fingerprint, environment_metadata,
)

APP_NAME = "Tashkent Property Price Command Center"

st.set_page_config(page_title=APP_NAME, page_icon="🏙️", layout="wide")

st.markdown("""
<style>
:root{--navy:#0f172a;--ink:#111827;--muted:#64748b;--teal:#0f766e;--blue:#2563eb;--line:#e2e8f0;--soft:#f8fafc;}
.block-container{padding-top:1rem;max-width:1360px}.main .block-container{padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#f8fafc,#eef6ff);border-right:1px solid #dbeafe}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 48%,#0f766e 100%);color:white;padding:2.2rem 2.4rem;border-radius:30px;box-shadow:0 24px 65px rgba(15,23,42,.24);margin-bottom:1.1rem}
.hero h1{font-size:2.55rem;letter-spacing:-.045em;margin:.25rem 0 1rem}.hero p{font-size:1.06rem;line-height:1.72;color:#e0f2fe;max-width:1120px}.kicker{font-size:.76rem;text-transform:uppercase;letter-spacing:.18em;font-weight:900;color:#a5f3fc}
.pill{display:inline-block;border:1px solid rgba(255,255,255,.28);background:rgba(255,255,255,.11);border-radius:999px;padding:.43rem .72rem;margin:.16rem .16rem .16rem 0;font-size:.82rem;font-weight:700}
.card{background:#fff;border:1px solid var(--line);border-radius:22px;padding:1.15rem 1.25rem;box-shadow:0 12px 30px rgba(15,23,42,.06);margin-bottom:1rem}.card h3{margin-top:0}.soft{background:linear-gradient(135deg,#ffffff,#eef9ff)}
.metric-card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1rem;min-height:102px;box-shadow:0 10px 25px rgba(15,23,42,.05)}.metric-label{font-size:.85rem;color:var(--muted)}.metric-value{font-size:1.55rem;font-weight:900;color:var(--ink);margin-top:.35rem}
.workflow{display:flex;gap:.55rem;flex-wrap:wrap}.step{background:#eef6ff;border:1px solid #bfdbfe;color:#1e3a8a;border-radius:14px;padding:.52rem .78rem;font-weight:800;font-size:.84rem}.muted{color:var(--muted);font-size:.92rem}.ok{color:#047857;font-weight:800}.warn{color:#b45309;font-weight:800}.bad{color:#b91c1c;font-weight:800}
.sidebar-chip{padding:.75rem .85rem;border-radius:18px;background:#fff;border:1px solid #dbeafe;margin:.6rem 0;box-shadow:0 8px 18px rgba(15,23,42,.04)}
</style>
""", unsafe_allow_html=True)


def money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:,.2f} bn UZS"
    return f"{x/1_000_000:,.1f} mln UZS"


def pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{100*x:.1f}%"


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(f"""
    <div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='muted'>{note}</div></div>
    """, unsafe_allow_html=True)


def hero():
    st.markdown("""
    <div class='hero'>
      <div class='kicker'>Pearson BTEC Level 6 · Applied AI Capstone</div>
      <h1>Tashkent Property Price Command Center</h1>
      <p>A professional real-estate analytics prototype for estimating apartment asking prices from OLX Uzbekistan listings. The system audits data quality, cleans noisy scraped fields, compares models against a baseline, explains price drivers and simulates valuation scenarios for transparent decision support.</p>
      <span class='pill'>Leakage-safe pipeline</span><span class='pill'>OLX market audit</span><span class='pill'>Regression model lab</span><span class='pill'>Scenario simulator</span><span class='pill'>Evidence export</span>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and cleaning real-estate data...")
def get_data(uploaded_bytes: bytes | None = None, uploaded_name: str | None = None):
    if uploaded_bytes is not None:
        raw = load_raw_data(uploaded_file=io.BytesIO(uploaded_bytes))
    else:
        path = find_dataset_path()
        raw = load_raw_data(path)
    clean = clean_real_estate_data(raw)
    return raw, clean


@st.cache_resource(show_spinner="Training models and calculating validation metrics...")
def train_cached(mode: str, max_rows: int | None, fingerprint: str):
    _, clean = get_data()
    return evaluate_models(clean, mode=mode, max_rows=max_rows)


def load_data_panel():
    st.sidebar.markdown("## 🏙️ Price Command Center")
    st.sidebar.markdown("PDP University · BTEC Level 6")
    with st.sidebar.expander("Dataset source", expanded=False):
        path = find_dataset_path()
        if path:
            st.success(f"Found: {path.name}")
        else:
            st.warning("No local dataset found. Upload an Excel file below.")
        uploaded = st.file_uploader("Upload OLX dataset", type=["xlsx", "xls", "csv", "pkl"])
        if uploaded:
            st.session_state["uploaded_dataset_bytes"] = uploaded.getvalue()
            st.session_state["uploaded_dataset_name"] = uploaded.name
            st.success(f"Uploaded: {uploaded.name}")
    try:
        if "uploaded_dataset_bytes" in st.session_state:
            raw, df = get_data(st.session_state["uploaded_dataset_bytes"], st.session_state.get("uploaded_dataset_name"))
        else:
            raw, df = get_data()
    except Exception as exc:
        st.error("Dataset could not be loaded.")
        st.info("Expected a valid OLX Excel workbook, not a Python script renamed as .xlsx. The safest file name is `olx_massive_real_estate.xlsx` in the repository root.")
        st.exception(exc)
        st.stop()
    return raw, df


raw_df, df = load_data_panel()
summary = market_summary(df)

pages = {
    "Operate": ["🏠 Overview", "📊 Data Audit", "🤖 Model Lab", "📈 Results"],
    "Use": ["💰 Price Prediction", "🎛️ Scenario Simulator", "🧭 Market Insights"],
    "Trust": ["🧪 Validation", "🔎 Explainability", "📋 Evidence Room"],
}
choices = []
st.sidebar.markdown("---")
for group, items in pages.items():
    st.sidebar.markdown(f"**{group.upper()}**")
    for item in items:
        choices.append(item)
page = st.sidebar.radio("Navigation", choices, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Rows after cleaning: {len(df):,}")
st.sidebar.caption(f"Dataset fingerprint: {dataset_fingerprint(df)}")
st.sidebar.caption("Target: asking price in UZS")

res = st.session_state.get("experiment")
best = res.get("best_model_name") if res else "Run training"
mae = money(float(res["leaderboard"].iloc[0]["MAE"])) if res else "—"
r2 = f"{float(res['leaderboard'].iloc[0]['R2']):.3f}" if res else "—"
st.markdown(f"""
<div class='card' style='position:sticky;top:.4rem;z-index:99;padding:.7rem 1rem;display:flex;gap:.75rem;flex-wrap:wrap;align-items:center'>
 <b>🏙️ Tashkent Property Price Command Center</b>
 <span class='step'>Experiment: {'Ready' if res else 'Not trained'}</span>
 <span class='step'>Best model: {best}</span>
 <span class='step'>MAE: {mae}</span>
 <span class='step'>R²: {r2}</span>
</div>
""", unsafe_allow_html=True)

if page == "🏠 Overview":
    hero()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Clean listings", f"{summary['rows']:,}", "after quality filters")
    with c2: metric_card("Districts", str(summary["districts"]), "market segments")
    with c3: metric_card("Median price", money(summary["median_price"]), "asking price")
    with c4: metric_card("Median price/m²", money(summary["median_price_per_sqm"]), "valuation anchor")
    with c5: metric_card("Features", str(summary["features"]), "model inputs")

    angle = st.radio("Presentation angle", ["Executive story", "Technical proof", "Viva defence"], horizontal=True)
    if angle == "Executive story":
        st.markdown("""<div class='card soft'><h3>From scraped listings to valuation support</h3><p>The project turns noisy OLX apartment listings into a structured valuation command center. It is designed for transparent exploration rather than blind price prediction.</p></div>""", unsafe_allow_html=True)
    elif angle == "Technical proof":
        st.markdown("""<div class='card soft'><h3>Methodological focus</h3><p>The pipeline removes identifiers, parses string prices and areas, avoids district target encoding leakage and evaluates models against a dummy median baseline using interpretable error metrics.</p></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='card soft'><h3>Defence position</h3><p>The prototype does not claim to predict transaction prices perfectly. It estimates asking prices and exposes uncertainty, outliers, feature drivers and practical limitations.</p></div>""", unsafe_allow_html=True)

    st.markdown("### Command workflow")
    st.markdown("""<div class='workflow'><span class='step'>1 · Data audit</span><span class='step'>2 · Clean pipeline</span><span class='step'>3 · Model lab</span><span class='step'>4 · Validation</span><span class='step'>5 · Prediction</span><span class='step'>6 · Scenario simulation</span><span class='step'>7 · Evidence export</span></div>""", unsafe_allow_html=True)

elif page == "📊 Data Audit":
    st.title("📊 Data Audit")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Raw rows", f"{len(raw_df):,}")
    with c2: metric_card("Clean rows", f"{len(df):,}")
    with c3: metric_card("Dropped rows", f"{len(raw_df)-len(df):,}")
    with c4: metric_card("Features", str(len(PREFERRED_FEATURES)))
    tab1, tab2, tab3, tab4 = st.tabs(["Target", "Missingness", "Outliers", "Data sample"])
    with tab1:
        fig = px.histogram(df, x=TARGET, nbins=60, title="Distribution of cleaned asking prices")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.histogram(df, x="price_per_sqm", nbins=60, title="Distribution of price per square metre")
        st.plotly_chart(fig2, width="stretch")
    with tab2:
        miss = raw_df.isna().mean().mul(100).sort_values(ascending=False).reset_index()
        miss.columns = ["column", "missing_percent"]
        st.dataframe(miss, width="stretch")
        st.download_button("Download missingness audit", miss.to_csv(index=False), "missingness_audit.csv")
    with tab3:
        st.markdown("The cleaning pipeline clips impossible room and floor values, filters extreme price-per-square-metre cases and keeps the modelling task focused on apartments with usable price and area fields.")
        st.dataframe(df[["district", TARGET, "total_area", "price_per_sqm", "number_of_rooms", "floor", "total_floors"]].head(100), width="stretch")
    with tab4:
        st.dataframe(df[PREFERRED_FEATURES + [TARGET, "price_per_sqm"]].head(100), width="stretch")
        st.download_button("Download cleaned sample", df.head(500).to_csv(index=False), "cleaned_sample.csv")

elif page == "🤖 Model Lab":
    st.title("🤖 Model Lab")
    mode = st.radio("Computation mode", ["quick", "full"], format_func=lambda x: "Quick · 3-fold CV" if x == "quick" else "Full · 5-fold CV", horizontal=True)
    max_rows = st.slider("Maximum training rows", 3000, min(30000, len(df)), min(12000, len(df)), 1000)
    st.info("Start with Quick mode. Use Full mode only after the app is stable, because Streamlit Cloud has limited resources.")
    if st.button("🚀 Train and evaluate candidate models", type="primary", width="stretch"):
        with st.spinner("Training candidate models..."):
            st.session_state["experiment"] = train_cached(mode, int(max_rows), dataset_fingerprint(df))
        st.success("Experiment completed. Open Results, Validation and Evidence Room.")
        st.rerun()
    if res:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Best model", res["best_model_name"])
        with c2: metric_card("MAE", money(float(res["leaderboard"].iloc[0]["MAE"])))
        with c3: metric_card("Within 20%", pct(float(res["leaderboard"].iloc[0]["Within_20pct"])))
        st.dataframe(res["leaderboard"], width="stretch")
        st.download_button("Download best model", save_model(res["best_model"]), "best_house_price_model.joblib")

elif page == "📈 Results":
    st.title("📈 Model Results")
    if not res:
        st.warning("Run Model Lab first.")
        st.stop()
    lb = res["leaderboard"].copy()
    tab1, tab2, tab3 = st.tabs(["Leaderboard", "Actual vs predicted", "Fold metrics"])
    with tab1:
        st.dataframe(lb, width="stretch")
        fig = px.bar(lb, x="Model", y="MAE", title="Lower MAE is better")
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download leaderboard", lb.to_csv(index=False), "leaderboard.csv")
    with tab2:
        pred = res["best_predictions"].copy()
        fig = px.scatter(pred, x="actual", y="predicted", title=f"Actual vs predicted — {res['best_model_name']}")
        lo = float(min(pred["actual"].min(), pred["predicted"].min()))
        hi = float(max(pred["actual"].max(), pred["predicted"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal"))
        st.plotly_chart(fig, width="stretch")
        residual = pred.assign(error=pred["predicted"] - pred["actual"], abs_error=lambda d: np.abs(d["error"]))
        fig2 = px.histogram(residual, x="error", nbins=60, title="Residual distribution")
        st.plotly_chart(fig2, width="stretch")
    with tab3:
        st.dataframe(res["fold_metrics"], width="stretch")

elif page == "💰 Price Prediction":
    st.title("💰 Price Prediction")
    if not res:
        st.warning("Run Model Lab first, then return to this page.")
        st.stop()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Property profile")
        vals = {}
        for col in ["district", "type_of_market", "house_type", "repairs", "furnished", "layout"]:
            options = sorted([str(x) for x in df[col].dropna().unique()])[:100]
            default = options.index(str(df[col].mode().iloc[0])) if str(df[col].mode().iloc[0]) in options else 0
            vals[col] = st.selectbox(col, options, index=default)
        vals["number_of_rooms"] = st.slider("number_of_rooms", 1, 8, int(df["number_of_rooms"].median()))
        vals["total_area"] = st.number_input("total_area m²", 15.0, 500.0, float(df["total_area"].median()), 1.0)
        vals["floor"] = st.slider("floor", 1, 60, int(df["floor"].median()))
        vals["total_floors"] = st.slider("total_floors", 1, 80, int(df["total_floors"].median()))
        vals["ceiling_height"] = st.number_input("ceiling_height m", 2.0, 6.0, float(df["ceiling_height"].median()) if df["ceiling_height"].notna().any() else 2.8, 0.1)
    with right:
        st.subheader("Estimate")
        all_vals = prepare_prediction_features(df, vals)
        if st.button("Estimate asking price", type="primary", width="stretch"):
            price = predict_single(res["best_model"], all_vals)
            sqm = price / max(float(all_vals["total_area"]), 1)
            metric_card("Estimated asking price", money(price), f"{money(sqm)} per m²")
            bench = district_benchmark(df)
            row = bench[bench["district"].astype(str) == str(all_vals["district"])]
            if not row.empty:
                median = float(row.iloc[0]["median_price"])
                delta = (price - median) / median
                st.info(f"Compared with the district median, this estimate is {delta:+.1%} different.")
            rec = pd.DataFrame([{**all_vals, "estimated_price_uzs": price, "estimated_price_per_sqm": sqm, "model": res["best_model_name"]}])
            st.download_button("Download prediction record", rec.to_csv(index=False), "prediction_record.csv")

elif page == "🎛️ Scenario Simulator":
    st.title("🎛️ Scenario Simulator")
    if not res:
        st.warning("Run Model Lab first.")
        st.stop()
    st.markdown("Simulate how the model reacts when valuation variables change. This is not causal proof; it is a decision-support sensitivity analysis.")
    base = {
        "district": str(df["district"].mode().iloc[0]),
        "type_of_market": str(df["type_of_market"].mode().iloc[0]),
        "house_type": str(df["house_type"].mode().iloc[0]),
        "repairs": str(df["repairs"].mode().iloc[0]),
        "furnished": str(df["furnished"].mode().iloc[0]),
        "layout": str(df["layout"].mode().iloc[0]),
        "number_of_rooms": int(df["number_of_rooms"].median()),
        "total_area": float(df["total_area"].median()),
        "floor": int(df["floor"].median()),
        "total_floors": int(df["total_floors"].median()),
        "ceiling_height": float(df["ceiling_height"].median()) if df["ceiling_height"].notna().any() else 2.8,
    }
    area_change = st.slider("Area change (%)", -30, 60, 0, 5)
    room_change = st.slider("Room count change", -2, 3, 0)
    floor_position = st.select_slider("Floor position scenario", options=["first floor", "middle floor", "top floor"], value="middle floor")
    repair_upgrade = st.selectbox("Repair scenario", sorted(df["repairs"].dropna().astype(str).unique())[:50])
    scenarios = []
    for name, modifier in [
        ("Current profile", {}),
        ("Larger area scenario", {"total_area": base["total_area"] * (1 + area_change / 100)}),
        ("Room layout scenario", {"number_of_rooms": max(1, base["number_of_rooms"] + room_change)}),
        ("Repair upgrade scenario", {"repairs": repair_upgrade}),
    ]:
        vals = base.copy(); vals.update(modifier)
        if floor_position == "first floor": vals["floor"] = 1
        if floor_position == "top floor": vals["floor"] = vals["total_floors"]
        if floor_position == "middle floor": vals["floor"] = max(2, int(vals["total_floors"] // 2))
        vals = prepare_prediction_features(df, vals)
        scenarios.append({"scenario": name, "estimated_price": predict_single(res["best_model"], vals)})
    scen = pd.DataFrame(scenarios)
    base_price = float(scen.iloc[0]["estimated_price"])
    scen["delta_vs_current"] = scen["estimated_price"] - base_price
    scen["delta_pct"] = scen["delta_vs_current"] / base_price
    st.dataframe(scen.assign(estimated_price=scen["estimated_price"].map(money), delta_vs_current=scen["delta_vs_current"].map(money), delta_pct=scen["delta_pct"].map(lambda x: f"{x:+.1%}")), width="stretch")
    fig = px.bar(scen, x="scenario", y="estimated_price", title="Scenario valuation comparison")
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download scenario analysis", scen.to_csv(index=False), "scenario_analysis.csv")

elif page == "🧭 Market Insights":
    st.title("🧭 Market Insights")
    bench = district_benchmark(df)
    st.dataframe(bench, width="stretch")
    fig = px.bar(bench.head(15), x="district", y="median_price_sqm", title="Top districts by median price per m²")
    st.plotly_chart(fig, width="stretch")
    fig2 = px.box(df, x="district", y=TARGET, title="Price distribution by district")
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, width="stretch")

elif page == "🧪 Validation":
    st.title("🧪 Validation and Reliability")
    if not res:
        st.warning("Run Model Lab first.")
        st.stop()
    pred = res["best_predictions"]
    ci = bootstrap_metric_ci(pred["actual"], pred["predicted"])
    st.dataframe(ci, width="stretch")
    fig = px.bar(ci, x="metric", y="estimate", error_y=ci["ci_high"]-ci["estimate"], error_y_minus=ci["estimate"]-ci["ci_low"], title="Bootstrap 95% confidence intervals")
    st.plotly_chart(fig, width="stretch")
    if st.button("Calculate learning curve", width="stretch"):
        lc = learning_curve_table(res["best_model"], df.sample(min(len(df), 12000), random_state=42))
        st.session_state["learning_curve"] = lc
    if "learning_curve" in st.session_state:
        lc = st.session_state["learning_curve"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=lc["train_size"], y=lc["train_mae"], mode="lines+markers", name="Training MAE"))
        fig2.add_trace(go.Scatter(x=lc["train_size"], y=lc["validation_mae"], mode="lines+markers", name="Validation MAE"))
        fig2.update_layout(title="Learning curve", xaxis_title="Training rows", yaxis_title="MAE")
        st.plotly_chart(fig2, width="stretch")

elif page == "🔎 Explainability":
    st.title("🔎 Explainability")
    if not res:
        st.warning("Run Model Lab first.")
        st.stop()
    X, y = split_xy(df)
    if st.button("Calculate permutation importance", type="primary"):
        with st.spinner("Calculating feature importance..."):
            st.session_state["importance"] = model_feature_importance(res["best_model"], X, y)
    if "importance" in st.session_state:
        imp = st.session_state["importance"]
        st.dataframe(imp, width="stretch")
        fig = px.bar(imp.head(20), x="importance_mae_increase", y="feature", orientation="h", title="Permutation importance: MAE increase when feature is shuffled")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download feature importance", imp.to_csv(index=False), "feature_importance.csv")

elif page == "📋 Evidence Room":
    st.title("📋 Evidence Room")
    st.markdown("""<div class='card soft'><h3>Evidence position</h3><p>The app is designed as a BTEC-style artefact: it contains a data pipeline, model comparison, validation diagnostics, explainability, scenario analysis and exportable evidence. It is stronger than a static notebook because the examiner can interact with the full workflow.</p></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Dataset fingerprint", dataset_fingerprint(df))
    with c2: metric_card("Environment", environment_metadata()["python"], "Python")
    with c3: metric_card("Current evidence", "Ready" if res else "Train first")
    if res:
        st.download_button("Download complete evidence pack", evidence_pack(res, df), "house_price_evidence_pack.zip", mime="application/zip", width="stretch")
        model_card = {
            "project": APP_NAME,
            "target": "apartment asking price in UZS",
            "best_model": res["best_model_name"],
            "rows_cleaned": len(df),
            "limitations": ["OLX asking prices are not final transaction prices", "scraped listings can contain noise", "local validation is required before business use"],
            "environment": environment_metadata(),
        }
        st.download_button("Download model card JSON", json.dumps(model_card, indent=2), "model_card.json")
    else:
        st.info("Run Model Lab to enable the full evidence export.")
