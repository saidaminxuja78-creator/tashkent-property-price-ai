from __future__ import annotations

from io import BytesIO

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ml_pipeline import (
    RANDOM_STATE,
    build_optuna_estimator,
    fit_default_model,
    get_models,
    get_preprocessor,
    load_data,
    prepare_features,
    split_columns,
    train_models,
)

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        .small-note {font-size: 0.88rem; opacity: 0.78;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner=False)
def get_cached_default_model(X_data: pd.DataFrame, y_data: np.ndarray):
    cat, num = split_columns(X_data)
    prep = get_preprocessor(cat, num)
    return fit_default_model(X_data, y_data, prep)


def metric_mean(bundle: dict, model_name: str, metric: str) -> float:
    return float(bundle["results"][model_name][metric]["mean"])


def metric_std(bundle: dict, model_name: str, metric: str) -> float:
    return float(bundle["results"][model_name][metric]["std"])


def format_result_table(bundle: dict) -> pd.DataFrame:
    rows = []
    sorted_models = sorted(
        bundle["results"],
        key=lambda name: metric_mean(bundle, name, "F1"),
        reverse=True,
    )
    for rank, name in enumerate(sorted_models, start=1):
        result = bundle["results"][name]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rows.append(
            {
                "Rank": f"{medal} {rank}".strip(),
                "Model": name,
                "Accuracy": f"{result['Accuracy']['mean']:.3f} ± {result['Accuracy']['std']:.3f}",
                "Precision": f"{result['Precision']['mean']:.3f} ± {result['Precision']['std']:.3f}",
                "Recall": f"{result['Recall']['mean']:.3f} ± {result['Recall']['std']:.3f}",
                "F1": f"{result['F1']['mean']:.3f} ± {result['F1']['std']:.3f}",
                "AUC-ROC": f"{result['AUC-ROC']['mean']:.3f} ± {result['AUC-ROC']['std']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def model_to_bytes(model) -> bytes:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def normalize_shap_explanation(explanation, feature_names):
    import shap

    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        values = values[:, :, class_index]
        if base_values.ndim == 2:
            base_values = base_values[:, class_index]
        elif base_values.ndim == 1 and len(base_values) > 1:
            base_values = np.repeat(base_values[class_index], values.shape[0])

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=list(feature_names),
    )


try:
    df = get_data()
except Exception as exc:
    st.error(f"Dataset yuklanmadi: {exc}")
    st.code(
        "Repository ichiga student-mat.csv faylini yoki "
        "data/student-mat.csv faylini joylashtiring."
    )
    st.stop()

# Early-warning setup: target=G3>=10, while G1/G2/G3 are excluded from X.
X, y = prepare_features(df, include_prior_grades=False)
cat_cols, num_cols = split_columns(X)
preprocessor = get_preprocessor(cat_cols, num_cols)

with st.sidebar:
    st.title("🎓 Student ML")
    st.markdown("**PDP University | 2026**")
    st.markdown("**Eltezorov Doriyorbek**")
    st.markdown("**Group: 22-305 | AI**")
    st.divider()
    page = st.radio(
        "📌 Bo'limlar",
        [
            "🏠 Bosh sahifa",
            "📊 EDA & Tahlil",
            "⚡ Optuna Optimization",
            "🤖 Model O'qitish",
            "📈 Natijalar",
            "🔬 SHAP Values",
            "🔍 Bashorat",
            "ℹ️ Model Card",
        ],
    )
    st.divider()
    st.caption(f"Dataset: {len(df)} qator")
    st.caption(f"Model featurelari: {X.shape[1]}")
    st.caption("Target: G3 ≥ 10 → Pass")
    st.caption("G1, G2 va G3 modelga berilmaydi")

bundle = st.session_state.get("training_bundle")

# ════════════════════════════════════════
# 🏠 BOSH SAHIFA
# ════════════════════════════════════════
if page == "🏠 Bosh sahifa":
    st.title("🎓 Machine Learning Model Optimization")
    st.subheader("Student Performance Prediction — Early Warning System")
    st.divider()

    best_name = bundle["best_model_name"] if bundle else "O'qitilmagan"
    best_f1 = metric_mean(bundle, best_name, "F1") if bundle else None
    best_auc = metric_mean(bundle, best_name, "AUC-ROC") if bundle else None

    cols = st.columns(5)
    cols[0].metric("👨‍🎓 O'quvchilar", len(df))
    cols[1].metric("📚 Features", X.shape[1])
    cols[2].metric("🤖 Modellar", len(get_models()))
    cols[3].metric("🏆 Best F1", f"{best_f1:.3f}" if best_f1 is not None else "—")
    cols[4].metric("📈 Best AUC", f"{best_auc:.3f}" if best_auc is not None else "—")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            """
            ### 📌 Loyiha nima qiladi?
            Ilova o'quvchining yakuniy natijasi **Pass/Fail** bo'lishini oldindan
            baholaydi. Modelga `G1`, `G2` va `G3` berilmaydi; shu sababli bu
            yakuniy bahoni takrorlovchi kalkulyator emas, balki erta xavf aniqlash tizimidir.

            ### 🔧 Metodlar
            - 6 ta klassifikatsiya modeli
            - 5-fold outer + 5-fold inner Nested Cross-Validation
            - GridSearchCV va Optuna
            - OOF ROC, Precision–Recall va Confusion Matrix
            - Haqiqiy Wilcoxon testi
            - SHAP global va local explanation
            - Real pipeline orqali individual bashorat
            """
        )
    with right:
        st.markdown("### 📊 Holat")
        if bundle:
            preview = format_result_table(bundle)[["Rank", "Model", "F1", "AUC-ROC"]]
            st.dataframe(preview, width="stretch", hide_index=True)
            st.success(f"Eng yaxshi model: {best_name}")
        else:
            st.info(
                "Natijalar hali hisoblanmagan. `🤖 Model O'qitish` bo'limiga o'tib "
                "Nested CV jarayonini ishga tushiring."
            )

# ════════════════════════════════════════
# 📊 EDA
# ════════════════════════════════════════
elif page == "📊 EDA & Tahlil":
    st.title("📊 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distribution", "🔗 Correlation", "📦 Boxplots", "📋 Dataset"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
            class_df = pd.DataFrame(
                {"Class": ["Fail", "Pass"], "Count": counts.values}
            )
            fig = px.bar(
                class_df,
                x="Class",
                y="Count",
                color="Class",
                text="Count",
                title="Pass vs Fail Distribution",
                color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")

        with col2:
            if "G3" in df.columns:
                fig = px.histogram(
                    df,
                    x="G3",
                    nbins=20,
                    title="Final Grade (G3) Distribution",
                )
                fig.add_vline(
                    x=10,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Pass threshold = 10",
                )
                st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            if "absences" in df.columns:
                fig = px.histogram(
                    df,
                    x="absences",
                    nbins=30,
                    title="Absences Distribution",
                )
                st.plotly_chart(fig, width="stretch")
        with col2:
            pie_df = pd.DataFrame(
                {
                    "Class": df["target"].map({0: "Fail", 1: "Pass"}),
                }
            )
            fig = px.pie(
                pie_df,
                names="Class",
                title="Class Balance",
                color="Class",
                color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.4,
            annot_kws={"size": 7},
            ax=ax,
        )
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        available = [
            col
            for col in [
                "studytime",
                "failures",
                "absences",
                "Medu",
                "Fedu",
                "famrel",
                "freetime",
            ]
            if col in df.columns
        ]
        feature = st.selectbox("Feature tanlang", available)
        y_axis = "G3" if "G3" in df.columns else "target"
        plot_df = df.copy()
        plot_df["Class"] = plot_df["target"].map({0: "Fail", 1: "Pass"})
        fig = px.box(
            plot_df,
            x=feature,
            y=y_axis,
            color="Class",
            title=f"{feature} vs {y_axis}",
            color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
        )
        st.plotly_chart(fig, width="stretch")

    with tab4:
        st.dataframe(df.head(50), width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        st.download_button(
            "📥 Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="student_dataset_with_target.csv",
            mime="text/csv",
            width="stretch",
        )

# ════════════════════════════════════════
# ⚡ OPTUNA
# ════════════════════════════════════════
elif page == "⚡ Optuna Optimization":
    st.title("⚡ Optuna Hyperparameter Optimization")
    st.divider()
    st.info(
        "Preprocessing har bir CV fold ichida bajariladi. Bu validation ma'lumotining "
        "oldindan ko'rilib qolishini oldini oladi."
    )

    model_choice = st.selectbox(
        "Model tanlang",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    )
    n_trials = st.slider("Trials soni", 10, 100, 30, 5)

    if st.button("🚀 Optuna Ishga Tushir", type="primary", width="stretch"):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        progress = st.progress(0)
        status = st.empty()

        def objective(trial):
            if model_choice == "Logistic Regression":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                    "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
                }
            elif model_choice == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 350, step=10),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }

            estimator = build_optuna_estimator(model_choice, params)
            pipe = Pipeline(
                [
                    ("pre", clone(preprocessor)),
                    ("clf", estimator),
                ]
            )
            scores = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="f1",
                n_jobs=1,
                error_score="raise",
            )
            return float(scores.mean())

        def callback(study, trial):
            done = trial.number + 1
            progress.progress(min(done / n_trials, 1.0))
            status.text(
                f"Trial {done}/{n_trials} — Best F1: {study.best_value:.4f}"
            )

        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(objective, n_trials=n_trials, callbacks=[callback], n_jobs=1)
        except Exception as exc:
            st.error(f"Optuna jarayonida xato: {exc}")
        else:
            best_estimator = build_optuna_estimator(model_choice, study.best_params)
            best_pipe = Pipeline(
                [
                    ("pre", clone(preprocessor)),
                    ("clf", best_estimator),
                ]
            )
            best_pipe.fit(X, y)
            st.session_state.setdefault("optuna_models", {})[model_choice] = best_pipe
            st.session_state["last_optuna_study"] = study

            progress.progress(1.0)
            status.empty()
            st.success("Optuna tugadi.")
            c1, c2 = st.columns(2)
            c1.metric("🏆 Eng yaxshi CV F1", f"{study.best_value:.4f}")
            c2.metric("Trials", len(study.trials))
            st.json(study.best_params)

            trials_df = study.trials_dataframe(
                attrs=("number", "value", "params", "state")
            )
            st.dataframe(trials_df, width="stretch", hide_index=True)

            history = pd.DataFrame(
                {
                    "Trial": [t.number + 1 for t in study.trials],
                    "F1": [t.value for t in study.trials],
                }
            )
            fig = px.line(
                history,
                x="Trial",
                y="F1",
                markers=True,
                title="Optuna Trial History",
            )
            fig.add_hline(
                y=study.best_value,
                line_dash="dash",
                annotation_text="Best",
            )
            st.plotly_chart(fig, width="stretch")

            try:
                importance = optuna.importance.get_param_importances(study)
                imp_df = pd.DataFrame(
                    {"Parameter": importance.keys(), "Importance": importance.values()}
                ).sort_values("Importance")
                fig = px.bar(
                    imp_df,
                    x="Importance",
                    y="Parameter",
                    orientation="h",
                    title="Hyperparameter Importance",
                )
                st.plotly_chart(fig, width="stretch")
            except Exception:
                st.caption("Parameter importance hisoblash uchun trials yetarli emas.")

# ════════════════════════════════════════
# 🤖 MODEL O'QITISH
# ════════════════════════════════════════
elif page == "🤖 Model O'qitish":
    st.title("🤖 Model O'qitish — Nested Cross-Validation")
    st.divider()
    st.warning(
        "6 model × 5 outer fold × 5 inner fold hisoblanadi. Streamlit Cloud'da "
        "bir necha daqiqa vaqt olishi mumkin."
    )

    if st.button("🚀 Modellarni O'qitish", type="primary", width="stretch"):
        progress = st.progress(0)
        status = st.empty()
        total_models = len(get_models())
        outer_splits = 5

        def update_progress(name, model_idx, models_count, fold_idx, folds_count):
            completed = (model_idx - 1) * folds_count + (fold_idx - 1)
            total = models_count * folds_count
            progress.progress(min(completed / total, 0.99))
            status.text(
                f"{name}: outer fold {fold_idx}/{folds_count} "
                f"({model_idx}/{models_count} model)"
            )

        try:
            trained = train_models(
                X,
                y,
                preprocessor,
                outer_splits=outer_splits,
                inner_splits=5,
                progress_callback=update_progress,
            )
        except Exception as exc:
            st.error(f"Model o'qitishda xato: {exc}")
        else:
            st.session_state["training_bundle"] = trained
            bundle = trained
            progress.progress(1.0)
            status.empty()
            st.success("Barcha modellar o'qitildi.")

    bundle = st.session_state.get("training_bundle")
    if bundle:
        best_name = bundle["best_model_name"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", best_name)
        c2.metric("Best F1", f"{metric_mean(bundle, best_name, 'F1'):.3f}")
        c3.metric("Best AUC", f"{metric_mean(bundle, best_name, 'AUC-ROC'):.3f}")

        st.subheader("Fold-by-Fold F1")
        fold_df = bundle["fold_metrics"]
        fig = px.line(
            fold_df,
            x="Fold",
            y="F1",
            color="Model",
            markers=True,
            title="F1 Score — Outer Folds",
        )
        st.plotly_chart(fig, width="stretch")

        best_model = bundle["best_pipes"][best_name]
        st.download_button(
            "📦 Eng yaxshi modelni yuklab olish",
            data=model_to_bytes(best_model),
            file_name="best_student_model.joblib",
            mime="application/octet-stream",
            width="stretch",
        )

# ════════════════════════════════════════
# 📈 NATIJALAR
# ════════════════════════════════════════
elif page == "📈 Natijalar":
    st.title("📈 Model Natijalari")
    st.divider()

    if not bundle:
        st.warning("Avval `🤖 Model O'qitish` bo'limida modellarni o'qiting.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏆 Leaderboard",
            "📊 Metrics",
            "📉 ROC & PR",
            "🧩 Confusion Matrix",
            "🧪 Wilcoxon",
        ]
    )

    model_names = list(bundle["results"].keys())
    best_name = bundle["best_model_name"]

    with tab1:
        leaderboard = format_result_table(bundle)
        st.dataframe(leaderboard, width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🥇 Best Model", best_name)
        c2.metric(
            "🏆 Best F1",
            f"{metric_mean(bundle, best_name, 'F1'):.3f} ± "
            f"{metric_std(bundle, best_name, 'F1'):.3f}",
        )
        c3.metric(
            "📈 Best AUC",
            f"{metric_mean(bundle, best_name, 'AUC-ROC'):.3f} ± "
            f"{metric_std(bundle, best_name, 'AUC-ROC'):.3f}",
        )

        export_df = leaderboard.copy()
        st.download_button(
            "📥 Natijalarni CSV yuklab olish",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="nested_cv_results.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab2:
        metric = st.selectbox(
            "Metric",
            ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"],
        )
        metric_df = pd.DataFrame(
            {
                "Model": model_names,
                "Mean": [metric_mean(bundle, name, metric) for name in model_names],
                "Std": [metric_std(bundle, name, metric) for name in model_names],
            }
        ).sort_values("Mean", ascending=False)

        fig = go.Figure(
            go.Bar(
                x=metric_df["Model"],
                y=metric_df["Mean"],
                error_y={"type": "data", "array": metric_df["Std"]},
                text=[f"{value:.3f}" for value in metric_df["Mean"]],
                textposition="outside",
            )
        )
        fig.update_layout(title=f"{metric} Taqqoslash", yaxis_range=[0, 1.08])
        st.plotly_chart(fig, width="stretch")

        st.subheader("Top 3 Radar Chart")
        categories = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        top3 = format_result_table(bundle)["Model"].head(3).tolist()
        radar = go.Figure()
        for name in top3:
            values = [metric_mean(bundle, name, item) for item in categories]
            radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=name,
                )
            )
        radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Top 3 Models",
        )
        st.plotly_chart(radar, width="stretch")

    with tab3:
        selected_models = st.multiselect(
            "Modellar",
            model_names,
            default=model_names,
        )
        roc_fig = go.Figure()
        pr_fig = go.Figure()

        for name in selected_models:
            pred_data = bundle["oof"][name]
            y_true = pred_data["y_true"]
            y_prob = pred_data["y_prob"]

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={roc_auc:.3f})",
                )
            )

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            pr_fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{name} (AUC={pr_auc:.3f})",
                )
            )

        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line={"dash": "dash"},
            )
        )
        roc_fig.update_layout(
            title="Out-of-Fold ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        pr_fig.update_layout(
            title="Out-of-Fold Precision–Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
        )
        st.plotly_chart(roc_fig, width="stretch")
        st.plotly_chart(pr_fig, width="stretch")

    with tab4:
        selected = st.selectbox("Model", model_names, index=model_names.index(best_name))
        threshold = st.slider("Probability threshold", 0.10, 0.90, 0.50, 0.05)
        pred_data = bundle["oof"][selected]
        y_pred_threshold = (pred_data["y_prob"] >= threshold).astype(int)
        cm = confusion_matrix(pred_data["y_true"], y_pred_threshold)

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Fail", "Predicted Pass"],
            y=["Actual Fail", "Actual Pass"],
            title=f"{selected} — Confusion Matrix",
            labels={"x": "Prediction", "y": "Actual", "color": "Count"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Thresholdni pasaytirish Fail xavfidagi o'quvchilarni ko'proq ushlashi "
            "mumkin, lekin false alarm sonini ham oshiradi."
        )

    with tab5:
        left, right = st.columns(2)
        model_a = left.selectbox("Model A", model_names, index=0)
        model_b_options = [name for name in model_names if name != model_a]
        model_b = right.selectbox("Model B", model_b_options, index=0)

        a_scores = bundle["f1_arrays"][model_a]
        b_scores = bundle["f1_arrays"][model_b]
        comparison_df = pd.DataFrame(
            {
                "Fold": np.arange(1, len(a_scores) + 1),
                model_a: a_scores,
                model_b: b_scores,
            }
        )
        st.dataframe(comparison_df, width="stretch", hide_index=True)

        try:
            statistic, p_value = wilcoxon(a_scores, b_scores)
            c1, c2 = st.columns(2)
            c1.metric("Wilcoxon statistic", f"{statistic:.4f}")
            c2.metric("p-value", f"{p_value:.4f}")
            if p_value < 0.05:
                st.success("Fold F1 natijalari orasida statistik ahamiyatli farq bor.")
            else:
                st.info(
                    "Statistik ahamiyatli farq aniqlanmadi. Bu modellar mutlaqo "
                    "bir xil degani emas; 5 ta fold test kuchini cheklaydi."
                )
        except ValueError as exc:
            st.warning(f"Wilcoxon hisoblanmadi: {exc}")

# ════════════════════════════════════════
# 🔬 SHAP
# ════════════════════════════════════════
elif page == "🔬 SHAP Values":
    st.title("🔬 SHAP Values — Model Tushuntirish")
    st.divider()
    st.info(
        "SHAP uchun Gradient Boosting ishlatiladi. Global grafiklar umumiy ta'sirni, "
        "waterfall esa bitta o'quvchi bashoratini tushuntiradi."
    )

    sample_size = st.slider("SHAP sample soni", 30, min(200, len(X)), min(100, len(X)), 10)

    if st.button("🔬 SHAP Hisoblash", type="primary", width="stretch"):
        import shap

        with st.spinner("SHAP hisoblanmoqda..."):
            if bundle and "Gradient Boosting" in bundle["best_pipes"]:
                shap_pipe = bundle["best_pipes"]["Gradient Boosting"]
            else:
                shap_pipe = Pipeline(
                    [
                        ("pre", clone(preprocessor)),
                        (
                            "clf",
                            GradientBoostingClassifier(
                                n_estimators=120,
                                learning_rate=0.05,
                                max_depth=2,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                )
                shap_pipe.fit(X, y)

            X_sample = X.sample(
                n=min(sample_size, len(X)), random_state=RANDOM_STATE
            )
            pre = shap_pipe.named_steps["pre"]
            clf = shap_pipe.named_steps["clf"]
            X_processed = np.asarray(pre.transform(X_sample))
            feature_names = pre.get_feature_names_out()

            background_size = min(100, len(X_processed))
            background = X_processed[:background_size]
            explainer = shap.Explainer(
                clf,
                background,
                feature_names=feature_names,
            )
            explanation = explainer(X_processed)
            explanation = normalize_shap_explanation(explanation, feature_names)

            st.session_state["shap_explanation"] = explanation
            st.session_state["shap_rows"] = X_sample.reset_index(drop=True)

        st.success("SHAP hisoblandi.")

    if "shap_explanation" in st.session_state:
        import shap

        explanation = st.session_state["shap_explanation"]
        shap_rows = st.session_state["shap_rows"]

        st.subheader("Global Feature Importance")
        plt.figure(figsize=(10, 7))
        shap.plots.bar(explanation, max_display=15, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("SHAP Beeswarm")
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Bitta o'quvchi uchun Local Explanation")
        row_number = st.slider(
            "Sample qatori",
            1,
            len(shap_rows),
            1,
        )
        st.dataframe(
            shap_rows.iloc[[row_number - 1]],
            width="stretch",
            hide_index=True,
        )
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(
            explanation[row_number - 1],
            max_display=15,
            show=False,
        )
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════
# 🔍 BASHORAT
# ════════════════════════════════════════
elif page == "🔍 Bashorat":
    st.title("🔍 O'quvchi Natijasini Bashorat Qilish")
    st.divider()

    if bundle:
        prediction_model_name = bundle["best_model_name"]
        prediction_model = bundle["best_pipes"][prediction_model_name]
        model_source = "Nested CV'dan keyingi eng yaxshi tuned pipeline"
    elif st.session_state.get("optuna_models"):
        prediction_model_name = next(iter(st.session_state["optuna_models"]))
        prediction_model = st.session_state["optuna_models"][prediction_model_name]
        model_source = "Optuna orqali o'qitilgan pipeline"
    else:
        prediction_model_name = "Logistic Regression (default)"
        prediction_model = get_cached_default_model(X, y)
        model_source = "Tezkor default pipeline; to'liq taqqoslash hali bajarilmagan"

    st.info(f"Model: **{prediction_model_name}** — {model_source}")
    threshold = st.slider("Qaror thresholdi", 0.10, 0.90, 0.50, 0.05)

    inputs = {}
    with st.form("prediction_form"):
        columns = st.columns(3)
        for idx, feature in enumerate(X.columns):
            container = columns[idx % 3]
            series = X[feature].dropna()

            with container:
                if feature in cat_cols:
                    options = series.unique().tolist()
                    mode = series.mode().iloc[0] if not series.mode().empty else options[0]
                    default_index = options.index(mode) if mode in options else 0
                    inputs[feature] = st.selectbox(
                        feature,
                        options,
                        index=default_index,
                        key=f"pred_{feature}",
                    )
                else:
                    numeric = pd.to_numeric(series, errors="coerce").dropna()
                    minimum = float(numeric.min())
                    maximum = float(numeric.max())
                    median = float(numeric.median())
                    is_integer = pd.api.types.is_integer_dtype(X[feature].dtype)
                    unique_values = sorted(numeric.unique().tolist())

                    if is_integer and len(unique_values) <= 20:
                        values = [int(value) for value in unique_values]
                        default = int(round(median))
                        default_index = (
                            values.index(default)
                            if default in values
                            else min(range(len(values)), key=lambda i: abs(values[i] - default))
                        )
                        inputs[feature] = st.selectbox(
                            feature,
                            values,
                            index=default_index,
                            key=f"pred_{feature}",
                        )
                    else:
                        step = 1.0 if is_integer else 0.1
                        value = int(round(median)) if is_integer else median
                        chosen = st.number_input(
                            feature,
                            min_value=int(minimum) if is_integer else minimum,
                            max_value=int(maximum) if is_integer else maximum,
                            value=value,
                            step=int(step) if is_integer else step,
                            key=f"pred_{feature}",
                        )
                        inputs[feature] = int(chosen) if is_integer else float(chosen)

        submitted = st.form_submit_button(
            "🎯 Bashorat Qilish",
            type="primary",
            width="stretch",
        )

    if submitted:
        student_df = pd.DataFrame([inputs], columns=X.columns)
        probability = float(prediction_model.predict_proba(student_df)[0, 1])
        prediction = int(probability >= threshold)

        st.divider()
        left, right = st.columns([1, 1.15])
        with left:
            if prediction == 1:
                st.success("✅ BASHORAT: PASS")
            else:
                st.error("❌ BASHORAT: FAIL RISK")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    title={"text": "Pass ehtimoli (%)"},
                    delta={"reference": threshold * 100},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, threshold * 100], "color": "#FADBD8"},
                            {"range": [threshold * 100, 100], "color": "#D5F5E3"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": threshold * 100,
                        },
                    },
                )
            )
            st.plotly_chart(gauge, width="stretch")

        with right:
            st.subheader("Kiritilgan ma'lumot")
            display_df = student_df.T.reset_index()
            display_df.columns = ["Feature", "Value"]
            st.dataframe(display_df, width="stretch", hide_index=True, height=430)

            if prediction == 0:
                st.warning(
                    "Bu natija jazo yoki yakuniy hukm emas. U o'quvchini qo'shimcha "
                    "qo'llab-quvvatlash uchun signal sifatida ishlatilishi kerak."
                )

# ════════════════════════════════════════
# ℹ️ MODEL CARD
# ════════════════════════════════════════
elif page == "ℹ️ Model Card":
    st.title("ℹ️ Model Card")
    st.divider()

    st.markdown(
        f"""
        ### Maqsad
        UCI Student Performance ma'lumotlari asosida o'quvchining `G3 ≥ 10`
        bo'lish ehtimolini baholash.

        ### Dataset
        - Qatorlar: **{len(df)}**
        - Model featurelari: **{X.shape[1]}**
        - Target: **0 = Fail, 1 = Pass**
        - Modeldan chiqarilgan ustunlar: **G1, G2, G3, target**

        ### Validatsiya
        - Outer CV: **5-fold StratifiedKFold**
        - Inner CV: **5-fold StratifiedKFold**
        - Asosiy tanlash metrikasi: **F1**
        - Qo'shimcha metrikalar: Accuracy, Precision, Recall, AUC-ROC

        ### Cheklovlar
        - Dataset kichik; natijalar boshqa universitetga avtomatik ko'chmaydi.
        - Model sababni isbotlamaydi; faqat statistik bog'lanishni o'rganadi.
        - Demografik featurelar fairness tekshiruvisiz qaror chiqarish uchun ishlatilmasligi kerak.
        - Bashorat o'qituvchi yoki ma'muriy qarorni almashtirmaydi.
        """
    )

    if bundle:
        best_name = bundle["best_model_name"]
        st.subheader("Joriy eng yaxshi model")
        st.write(f"**Model:** {best_name}")
        st.write(f"**Best parameters:** {bundle['results'][best_name]['best_params']}")
        st.write(
            f"**F1:** {metric_mean(bundle, best_name, 'F1'):.3f} ± "
            f"{metric_std(bundle, best_name, 'F1'):.3f}"
        )
        st.write(
            f"**AUC-ROC:** {metric_mean(bundle, best_name, 'AUC-ROC'):.3f} ± "
            f"{metric_std(bundle, best_name, 'AUC-ROC'):.3f}"
        )
    else:
        st.info("Joriy trening natijasi mavjud emas.")
