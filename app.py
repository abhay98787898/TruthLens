
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "final_svm_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"
REPORTS_DIR = ROOT / "reports"

st.set_page_config(
    page_title="TruthLens | Fake News Research",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263244;
}
.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}
.metric-card {
    background: #111c2e;
    border: 1px solid #26354b;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 10px;
}
.hero {
    padding: 26px;
    border: 1px solid #26354b;
    border-radius: 18px;
    background: linear-gradient(120deg, #13233a, #111827);
    margin-bottom: 22px;
}
.small-muted {
    color: #9ca3af;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model missing. Run: python scripts/train_final_model.py"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


st.sidebar.title("🔎 TruthLens")
st.sidebar.caption("Explainable Fake News Research")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Analyze News", "Model Insights", "About Research"],
)
st.sidebar.divider()
st.sidebar.caption("Local research prototype • WELFake dataset")


def read_report(filename):
    path = REPORTS_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()



if page == "Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>TruthLens Research Dashboard</h1>
      <p>Explainable fake-news classification with group-aware evaluation.</p>
      <span class="small-muted">TF-IDF · Linear SVM · Multi-seed robustness</span>
    </div>
    """, unsafe_allow_html=True)

    # Existing reports
    baseline = read_report("baseline_metrics.csv")
    group = read_report("text_group_split_metrics.csv")
    ablation = read_report("cue_ablation_metrics.csv")

    # Corrected multi-seed reports
    multiseed = read_report(
        "multiseed_robustness_corrected_summary.csv"
    )
    per_seed = read_report(
        "multiseed_robustness_corrected_per_seed.csv"
    )
    paired = read_report(
        "multiseed_f1_paired_differences_corrected.csv"
    )

    st.subheader("Research overview")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Final model training rows",
        f"{metadata.get('training_rows', 0):,}"
    )
    c2.metric("Multi-seed evaluation seeds", "3")
    c3.metric("Near-duplicate group overlap", "0 shared groups")

    st.caption(
        "Training-row count refers to the separately trained demo model. "
        "Evaluation metrics below come from group-aware research experiments."
    )

    st.divider()
    st.subheader("Multi-seed robustness evaluation")

    if not multiseed.empty:
        st.caption(
            "Values show mean ± sample standard deviation across seeds 7, 21 and 42."
        )

        display = multiseed.copy()
        display["variant"] = display["variant"].replace({
            "baseline_original_text": "Original text",
            "source_and_boilerplate_cues_removed": "Source/boilerplate cues removed",
        })

        metric_names = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 score"),
            ("roc_auc", "ROC-AUC"),
        ]

        for metric, label in metric_names:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in display.columns and std_col in display.columns:
                display[label] = display.apply(
                    lambda row: (
                        f"{float(row[mean_col]):.4f} ± "
                        f"{float(row[std_col]):.4f}"
                    ),
                    axis=1,
                )

        visible_columns = ["variant"] + [
            label for _, label in metric_names
            if label in display.columns
        ]

        st.dataframe(
            display[visible_columns],
            width="stretch",
            hide_index=True,
        )

        chart_data = multiseed.copy()
        chart_data["variant"] = chart_data["variant"].replace({
            "baseline_original_text": "Original text",
            "source_and_boilerplate_cues_removed": "Cues removed",
        })

        chart = px.bar(
            chart_data,
            x="variant",
            y="f1_mean",
            error_y="f1_std",
            title="Mean F1 score across three seeds",
            labels={
                "variant": "Experiment",
                "f1_mean": "Mean F1 score",
            },
            template="plotly_dark",
        )
        st.plotly_chart(chart, width="stretch")

        baseline_row = multiseed[
            multiseed["variant"] == "baseline_original_text"
        ]
        cleaned_row = multiseed[
            multiseed["variant"]
            == "source_and_boilerplate_cues_removed"
        ]

        if not baseline_row.empty and not cleaned_row.empty:
            base_f1 = float(baseline_row.iloc[0]["f1_mean"])
            clean_f1 = float(cleaned_row.iloc[0]["f1_mean"])
            difference_pp = (base_f1 - clean_f1) * 100

            st.metric(
                "Mean F1 difference",
                f"{difference_pp:.2f} percentage points",
                help=(
                    "Original text minus source/boilerplate cue-removed text. "
                    "This is an observed association, not proof of leakage or causation."
                ),
            )
    else:
        st.warning(
            "Corrected multi-seed summary not found. "
            "Check reports/multiseed_robustness_corrected_summary.csv."
        )

    if not per_seed.empty:
        with st.expander("View individual seed results"):
            st.dataframe(
                per_seed,
                width="stretch",
                hide_index=True,
            )

    if not paired.empty:
        with st.expander("View paired F1 differences"):
            st.dataframe(
                paired,
                width="stretch",
                hide_index=True,
            )

    st.divider()
    st.subheader("Earlier research reports")

    with st.expander("Baseline model experiments"):
        if not baseline.empty:
            st.dataframe(baseline, width="stretch", hide_index=True)
        else:
            st.info("Baseline report not available.")

    with st.expander("Article-body group split"):
        if not group.empty:
            st.dataframe(group, width="stretch", hide_index=True)
        else:
            st.info("Article-body group-split report not available.")

    with st.expander("Source-cue ablation"):
        if not ablation.empty:
            st.dataframe(ablation, width="stretch", hide_index=True)
        else:
            st.info("Source-cue ablation report not available.")

    st.divider()
    st.subheader("Research limitations")

    st.markdown("""
    - Evaluation uses three different splits of the same WELFake dataset;
      it is not external-dataset validation.
    - Group-aware splitting prevents identified groups from crossing
      train and test sets, but does not guarantee publisher, topic or
      temporal independence.
    - MinHash LSH is approximate and may miss near duplicates.
    - Source-cue removal uses a manually selected list. The F1 difference
      does not prove that the cues caused data leakage.
    - LinearSVC decision scores are not calibrated probabilities.
    """)

    st.caption(
        "The saved prediction model is trained on the cleaned dataset for "
        "demonstration. Its performance should not be presented as independent "
        "test performance from the multi-seed experiments."
    )



elif page == "Analyze News":
    st.markdown("""
    <div class="hero">
      <h1>Analyze a News Article</h1>
      <p>Enter a headline and article body to inspect the model prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    title = st.text_input(
        "News headline",
        placeholder="Enter the headline...",
    )
    body = st.text_area(
        "Article text",
        placeholder="Paste the news article here...",
        height=250,
    )

    if st.button("Analyze News", type="primary", width="stretch"):
        if not title.strip() and not body.strip():
            st.warning("Please enter a headline or article text.")
        elif not body.strip():
            st.warning(
                "Please enter the article body. This model was trained "
                "on headline plus article text; empty-body predictions "
                "may be unreliable."
            )
        else:
            content = (title + " " + body).lower().strip()
            content = " ".join(content.split())

            prediction = int(model.predict([content])[0])
            class_name = "Real" if prediction == 1 else "Fake"
            decision_score = float(model.decision_function([content])[0])

            st.divider()
            st.subheader("Prediction")

            if prediction == 1:
                st.success(f"Model classification: {class_name} news")
            else:
                st.error(f"Model classification: {class_name} news")

            left, right = st.columns(2)
            left.metric("Predicted class", class_name)
            right.metric("Decision score", f"{decision_score:.4f}")

            st.info(
                "Decision score is not a calibrated probability or a "
                "confidence percentage. It measures the model's signed "
                "distance from its decision boundary."
            )

            st.warning(
                "This is an ML classification, not fact verification. "
                "Check the original source, date, evidence and independent "
                "reliable reporting before drawing conclusions."
            )

            vectorizer = model.named_steps["tfidf"]
            classifier = model.named_steps["classifier"]
            vector = vectorizer.transform([content])
            feature_names = vectorizer.get_feature_names_out()
            active_indices = vector.nonzero()[1]

            contributions = []
            for idx in active_indices:
                value = float(vector[0, idx] * classifier.coef_[0, idx])
                if value != 0:
                    contributions.append(
                        (feature_names[idx], value)
                    )

            contributions.sort(
                key=lambda item: abs(item[1]), reverse=True
            )

            st.subheader("Words influencing this prediction")
            if contributions:
                explanation = pd.DataFrame(
                    contributions[:15],
                    columns=["Feature", "Contribution"],
                )
                explanation["Direction"] = explanation[
                    "Contribution"
                ].apply(
                    lambda value: (
                        "Toward Real" if value > 0 else "Toward Fake"
                    )
                )

                fig = px.bar(
                    explanation.sort_values("Contribution"),
                    x="Contribution",
                    y="Feature",
                    color="Contribution",
                    color_continuous_midpoint=0,
                    orientation="h",
                    template="plotly_dark",
                    title="Linear SVM feature contributions",
                )
                st.plotly_chart(fig, width="stretch")
                st.dataframe(
                    explanation,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.write(
                    "No active weighted vocabulary features were found "
                    "for this input."
                )

            st.caption(
                "Feature contributions explain the trained model's "
                "decision, not whether each statement is factually true."
            )



elif page == "Model Insights":
    st.markdown("""
    <div class="hero">
      <h1>Model Insights</h1>
      <p>Compare evaluation settings, test robustness, and inspect influential features.</p>
    </div>
    """, unsafe_allow_html=True)

    # Corrected multi-seed research reports
    multiseed = read_report(
        "multiseed_robustness_corrected_summary.csv"
    )
    per_seed = read_report(
        "multiseed_robustness_corrected_per_seed.csv"
    )
    paired = read_report(
        "multiseed_f1_paired_differences_corrected.csv"
    )

    # Existing reports: retained for comparison and error analysis
    ablation = read_report("cue_ablation_metrics.csv")
    features = read_report("top_logistic_features.csv")
    errors = read_report("misclassified_examples.csv")

    st.subheader("Multi-seed robustness")

    if not multiseed.empty:
        st.caption(
            "Mean ± sample standard deviation across seeds 7, 21 and 42. "
            "These are group-aware experiments on the same WELFake dataset."
        )

        display = multiseed.copy()
        display["variant"] = display["variant"].replace({
            "baseline_original_text": "Original text",
            "source_and_boilerplate_cues_removed":
                "Source/boilerplate cues removed",
        })

        metric_names = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 score"),
            ("roc_auc", "ROC-AUC"),
        ]

        for metric, label in metric_names:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in display.columns and std_col in display.columns:
                display[label] = display.apply(
                    lambda row: (
                        f"{float(row[mean_col]):.4f} ± "
                        f"{float(row[std_col]):.4f}"
                    ),
                    axis=1,
                )

        visible_columns = ["variant"] + [
            label for _, label in metric_names
            if label in display.columns
        ]

        st.dataframe(
            display[visible_columns],
            width="stretch",
            hide_index=True,
        )

        baseline_rows = multiseed[
            multiseed["variant"] == "baseline_original_text"
        ]
        removed_rows = multiseed[
            multiseed["variant"]
            == "source_and_boilerplate_cues_removed"
        ]

        if not baseline_rows.empty and not removed_rows.empty:
            baseline_f1 = float(
                baseline_rows.iloc[0]["f1_mean"]
            )
            removed_f1 = float(
                removed_rows.iloc[0]["f1_mean"]
            )
            difference = (baseline_f1 - removed_f1) * 100

            c1, c2 = st.columns(2)
            c1.metric(
                "Original text mean F1",
                f"{baseline_f1 * 100:.2f}%"
            )
            c2.metric(
                "Mean F1 difference",
                f"{difference:.2f} percentage points"
            )

            st.info(
                "Interpretation: the original-text experiment performed "
                "better across the three tested seeds. This observed "
                "difference does not, by itself, prove data leakage "
                "or causation."
            )
    else:
        st.warning(
            "Corrected multi-seed summary was not found. "
            "Check the reports folder and CSV filename."
        )

    if not per_seed.empty:
        with st.expander("View individual seed results"):
            st.dataframe(
                per_seed,
                width="stretch",
                hide_index=True,
            )

    if not paired.empty:
        with st.expander("View paired F1 differences"):
            st.dataframe(
                paired,
                width="stretch",
                hide_index=True,
            )

    st.divider()
    st.subheader("Earlier cue-ablation comparison")

    if not ablation.empty:
        st.caption(
            "This is a separate earlier experiment comparing Logistic "
            "Regression and Linear SVM. Its evaluation setup differs "
            "from the corrected three-seed robustness experiment above."
        )
        st.dataframe(
            ablation,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Earlier cue-ablation results are not available.")

    st.divider()
    st.subheader("Top Logistic Regression features")

    if not features.empty:
        st.caption(
            "These coefficients belong to the separate Logistic "
            "Regression comparison model, not the final Linear SVM."
        )
        st.dataframe(
            features.head(30),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Feature report is not available.")

    st.divider()
    st.subheader("Sample misclassified records")

    if not errors.empty:
        st.caption(
            "Examples of errors from the saved research report. "
            "Inspect them as individual cases, not as a complete "
            "measure of model performance."
        )
        st.dataframe(
            errors.head(20),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Misclassified-record report is not available.")

    st.divider()
    st.subheader("Research limitations")

    st.markdown("""
    - All three seeds use different splits of the same WELFake dataset;
      this is not external-dataset validation.
    - Group-aware splitting prevents identified groups from crossing
      train and test sets, but does not guarantee publisher, topic,
      or temporal independence.
    - MinHash LSH is approximate and may miss near-duplicate articles.
    - The source/boilerplate cue list was selected manually. The observed
      metric difference does not prove that these cues caused data leakage.
    - Logistic Regression coefficients are from a separate comparison
      model and should not be presented as explanations of the final SVM.
    - LinearSVC decision scores are not calibrated probabilities.
    - The model does not verify news claims against live sources.
    """)



else:
    st.title("About TruthLens")
    st.write(
        "TruthLens is an academic research prototype for fake-news "
        "classification using machine learning and text features."
    )

    st.subheader("Methodology")
    st.markdown("""
    - Dataset: WELFake
    - Text representation: TF-IDF word unigrams and bigrams
    - Final classifier: Linear Support Vector Machine
    - Evaluation: stratified and group-based experiments
    - Explainability: weighted feature contributions from the linear model
    """)

    st.subheader("Label mapping")
    st.write("Label 0 = Fake; Label 1 = Real.")

    st.subheader("Research limitations")
    st.markdown("""
    - Dataset labels and source distribution may introduce bias.
    - A high test score does not prove generalization to future news.
    - The model does not verify claims against live sources.
    - Its decision score is not a probability.
    - Explanations show model associations, not causal evidence.
    """)
