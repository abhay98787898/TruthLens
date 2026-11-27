from datetime import date
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

INK = "#3b2f2a"
BRICK = "#b5532f"
GREEN = "#5f7a61"
GREY = "#8a7766"

st.set_page_config(
    page_title="TruthLens",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

:root {
    --cocoa: #3b2f2a;
    --grey: #8a7766;
    --cream: #fbf5ea;
    --sand: #efe2cc;
    --line: #e0d0b4;
    --terracotta: #b5532f;
    --sage: #5f7a61;
    --shadow: 0 3px 14px rgba(120, 85, 40, 0.10);
}

.stApp {
    background: linear-gradient(180deg, #f6ecda 0%, #f1e3cb 100%);
    color: var(--cocoa);
}
[data-testid="stHeader"] { background: transparent; }
.stApp p, .stApp li, .stApp label, .stApp span,
[data-testid="stMarkdownContainer"] { color: var(--cocoa); }

.block-container { max-width: 1080px; padding-top: 2rem; }
footer { visibility: hidden; }

h1, h2, h3, .masthead h1 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--cocoa) !important;
    letter-spacing: -0.01em;
}

.masthead { background: var(--cream); border: 1px solid var(--line);
            border-radius: 18px; box-shadow: var(--shadow);
            padding: 16px 28px 20px 28px; margin-bottom: 22px; }
.masthead .dateline { font-size: 0.75rem; letter-spacing: 0.08em;
                      text-transform: uppercase; color: var(--grey);
                      display: flex; justify-content: space-between; }
.masthead h1 { font-size: 3.2rem; margin: 6px 0 2px 0; padding: 0; line-height: 1.1; }
.masthead .tagline { font-family: 'Source Serif 4', Georgia, serif;
                     font-style: italic; color: var(--grey); font-size: 1.05rem; }

.stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: 'Source Serif 4', Georgia, serif;
                               font-size: 1.05rem; padding-left: 2px; padding-right: 2px; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--terracotta); }

.figures { display: grid; grid-template-columns: repeat(3, 1fr);
           gap: 14px; margin: 16px 0 8px 0; }
.figures div { background: var(--cream); border: 1px solid var(--line);
               border-radius: 14px; box-shadow: var(--shadow);
               padding: 16px 20px; }
.figures .num { display: block; font-family: 'Source Serif 4', Georgia, serif;
                font-size: 2.1rem; font-weight: 600; color: var(--terracotta); }
.figures .lbl { font-size: 0.85rem; color: var(--grey); }

.verdict { background: var(--cream); border: 1px solid var(--line);
           border-left: 8px solid var(--grey); border-radius: 14px;
           box-shadow: var(--shadow); padding: 16px 22px; margin: 12px 0 16px 0; }
.verdict.fake { border-left-color: var(--terracotta); }
.verdict.real { border-left-color: var(--sage); }
.verdict .small { font-size: 0.75rem; letter-spacing: 0.08em;
                  text-transform: uppercase; color: var(--grey); }
.verdict .word { font-family: 'Source Serif 4', Georgia, serif;
                 font-size: 2.4rem; font-weight: 700; line-height: 1.15; }
.verdict.fake .word { color: var(--terracotta); }
.verdict.real .word { color: var(--sage); }
.verdict .score { font-family: monospace; color: var(--grey); }

.st-key-input_card { background: var(--cream); border: 1px solid var(--line) !important;
                     border-radius: 16px; box-shadow: var(--shadow); padding: 8px 12px; }
[data-testid="stExpander"] { background: var(--cream); border: 1px solid var(--line);
                             border-radius: 12px; }
[data-testid="stExpander"] details { border: none; }

div[data-baseweb="input"], div[data-baseweb="textarea"] {
    border-radius: 10px; background: #fffaf0; }
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
    background: #fffaf0; color: var(--cocoa); }
.stButton > button { border-radius: 999px; font-weight: 600; padding: 0.5rem 1.6rem;
                     background: var(--terracotta); border: none; color: #fff7ea; }
.stButton > button:hover { background: #9c4526; color: #fff7ea; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run: python scripts/train_final_model.py"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def read_report(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ---------- helpers ----------

VARIANT_NAMES = {
    "baseline_original_text": "Original text",
    "source_and_boilerplate_cues_removed": "Source/boilerplate cues removed",
}
METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 score"),
    ("roc_auc", "ROC-AUC"),
]


def show_table(df):
    st.dataframe(df, width="stretch", hide_index=True)


def summary_table(df):
    out = pd.DataFrame({"Setup": df["variant"].replace(VARIANT_NAMES)})
    for key, label in METRICS:
        mean_col, std_col = f"{key}_mean", f"{key}_std"
        if mean_col in df.columns and std_col in df.columns:
            out[label] = [
                f"{m:.4f} ± {s:.4f}" for m, s in zip(df[mean_col], df[std_col])
            ]
    return out


def f1_pair(df):
    base = df.loc[df["variant"] == "baseline_original_text", "f1_mean"]
    cleaned = df.loc[
        df["variant"] == "source_and_boilerplate_cues_removed", "f1_mean"
    ]
    if base.empty or cleaned.empty:
        return None
    return float(base.iloc[0]), float(cleaned.iloc[0])


def style_fig(fig):
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def section_note(text):
    st.markdown(f"<span style='color:{GREY}'>{text}</span>",
                unsafe_allow_html=True)


# ---------- masthead ----------

st.markdown(f"""
<div class="masthead">
  <div class="dateline">
    <span>TF-IDF + Linear SVM &nbsp;|&nbsp; WELFake dataset</span>
    <span>{date.today().strftime("%A, %d %B %Y")}</span>
  </div>
  <h1>TruthLens</h1>
  <div class="tagline">A fake-news classifier, and a look at how far its
  scores can really be trusted.</div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_check, tab_eval, tab_notes = st.tabs(
    ["Overview", "Check an article", "Evaluation details", "Notes and limits"]
)


# ---------- overview ----------

with tab_overview:
    st.write(
        "TruthLens takes a headline and an article body and labels the text "
        "Fake or Real using TF-IDF features and a linear SVM. The classifier "
        "itself is simple. Most of the work in this project went into "
        "evaluation: removing duplicates, splitting by groups so similar "
        "articles don't land on both sides, and testing what happens when "
        "source-specific wording is stripped out."
    )

    rows = metadata.get("training_rows", 0)
    st.markdown(f"""
    <div class="figures">
      <div><span class="num">{rows:,}</span>
           <span class="lbl">rows used to train the demo model</span></div>
      <div><span class="num">3</span>
           <span class="lbl">random seeds in the robustness runs</span></div>
      <div><span class="num">0</span>
           <span class="lbl">near-duplicate groups shared by train and test</span></div>
    </div>
    """, unsafe_allow_html=True)
    section_note(
        "The row count is for the separately trained demo model. The scores "
        "below come from the group-aware experiments."
    )

    st.subheader("Main result: three seeds")
    multiseed = read_report("multiseed_robustness_corrected_summary.csv")

    if multiseed.empty:
        st.warning(
            "reports/multiseed_robustness_corrected_summary.csv is missing. "
            "Run scripts/research_multiseed_robustness.py to generate it."
        )
    else:
        section_note("Mean ± sample standard deviation over seeds 7, 21 and 42.")
        show_table(summary_table(multiseed))

        chart_df = multiseed.copy()
        chart_df["variant"] = chart_df["variant"].replace(VARIANT_NAMES)
        fig = px.scatter(
            chart_df, x="variant", y="f1_mean", error_y="f1_std",
            labels={"variant": "", "f1_mean": "Mean F1"},
            title="Mean F1 with one standard deviation",
        )
        fig.update_traces(marker=dict(size=14, color=BRICK),
                          error_y=dict(color=INK, thickness=1.5))
        st.plotly_chart(style_fig(fig), width="stretch")

        pair = f1_pair(multiseed)
        if pair:
            gap = (pair[0] - pair[1]) * 100
            st.write(
                f"F1 drops by **{gap:.2f} percentage points** when the "
                "source and boilerplate cues are removed. That is an observed "
                "difference only; it does not prove leakage or say what caused it."
            )

    st.subheader("Earlier experiments")
    for title, fname in [
        ("Baseline models", "baseline_metrics.csv"),
        ("Article-body group split", "text_group_split_metrics.csv"),
        ("Source-cue ablation", "cue_ablation_metrics.csv"),
    ]:
        with st.expander(title):
            df = read_report(fname)
            if df.empty:
                st.info(f"{fname} not found in reports/.")
            else:
                show_table(df)


# ---------- check an article ----------

with tab_check:
    st.subheader("Check an article")
    st.write("Paste a headline and the article text. The model was trained on "
             "both together, so please fill in both.")

    with st.container(border=True, key="input_card"):
        title = st.text_input("Headline")
        body = st.text_area("Article text", height=240)
        run_clicked = st.button("Run the classifier", type="primary")

    if run_clicked:
        if not title.strip() and not body.strip():
            st.warning("Enter a headline and the article text first.")
        elif not body.strip():
            st.warning(
                "The article body is empty. The model was trained on "
                "headline + body, so a headline alone can give unreliable output."
            )
        else:
            content = " ".join((title + " " + body).lower().split())

            prediction = int(model.predict([content])[0])
            score = float(model.decision_function([content])[0])
            label = "Real" if prediction == 1 else "Fake"
            css = "real" if prediction == 1 else "fake"

            st.markdown(f"""
            <div class="verdict {css}">
              <div class="small">Model output</div>
              <div class="word">{label}</div>
              <div class="score">decision score {score:+.4f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.write(
                "The decision score is the signed distance from the SVM's "
                "boundary. It is not a probability or a confidence percentage. "
                "This is a text-pattern classification, not fact-checking, so "
                "look at the original source and independent reporting too."
            )

            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["classifier"]
            vec = tfidf.transform([content])
            names = tfidf.get_feature_names_out()

            contributions = []
            for idx in vec.nonzero()[1]:
                value = float(vec[0, idx] * clf.coef_[0, idx])
                if value != 0:
                    contributions.append((names[idx], value))
            contributions.sort(key=lambda item: abs(item[1]), reverse=True)

            st.subheader("Words that pushed the decision")
            if not contributions:
                st.write("None of the words in this text are in the model's vocabulary.")
            else:
                top = pd.DataFrame(contributions[:15],
                                   columns=["Feature", "Contribution"])
                top["Direction"] = top["Contribution"].apply(
                    lambda v: "Toward Real" if v > 0 else "Toward Fake"
                )
                fig = px.bar(
                    top.sort_values("Contribution"),
                    x="Contribution", y="Feature", color="Direction",
                    orientation="h",
                    color_discrete_map={"Toward Real": GREEN,
                                        "Toward Fake": BRICK},
                )
                fig.update_layout(legend_title_text="", yaxis_title="")
                st.plotly_chart(style_fig(fig), width="stretch")
                show_table(top)
                section_note(
                    "These contributions describe how the trained model "
                    "decided, not whether a statement is true."
                )


# ---------- evaluation details ----------

with tab_eval:
    st.subheader("Per-seed results")
    per_seed = read_report("multiseed_robustness_corrected_per_seed.csv")
    paired = read_report("multiseed_f1_paired_differences_corrected.csv")

    if per_seed.empty:
        st.info("Per-seed report not found in reports/.")
    else:
        show_table(per_seed)

    st.subheader("Paired F1 differences")
    if paired.empty:
        st.info("Paired-difference report not found in reports/.")
    else:
        show_table(paired)

    st.subheader("Earlier cue-ablation run")
    ablation = read_report("cue_ablation_metrics.csv")
    if ablation.empty:
        st.info("cue_ablation_metrics.csv not found in reports/.")
    else:
        section_note(
            "A separate, earlier experiment (Logistic Regression vs Linear SVM). "
            "Its setup differs from the three-seed runs, so don't compare the "
            "numbers directly."
        )
        show_table(ablation)

    st.subheader("Top Logistic Regression features")
    features = read_report("top_logistic_features.csv")
    if features.empty:
        st.info("top_logistic_features.csv not found in reports/.")
    else:
        section_note(
            "These come from the comparison Logistic Regression model, not "
            "from the final SVM."
        )
        show_table(features.head(30))

    st.subheader("Misclassified examples")
    errors = read_report("misclassified_examples.csv")
    if errors.empty:
        st.info("misclassified_examples.csv not found in reports/.")
    else:
        section_note("Individual cases from the saved report, not an overall error rate.")
        show_table(errors.head(20))


# ---------- notes ----------

with tab_notes:
    st.subheader("Method")
    st.markdown("""
    - Dataset: WELFake (headline + body, labelled Fake or Real)
    - Cleaning: empty texts and exact duplicates removed
    - Features: TF-IDF on word unigrams and bigrams
    - Final model: Linear Support Vector Machine
    - Evaluation: stratified splits, then group-based splits using near-duplicate groups
    - Explanation: feature weight times TF-IDF value, per word
    - Labels: 0 = Fake, 1 = Real
    """)

    st.subheader("Limitations")
    st.markdown("""
    - Every split comes from the same WELFake dataset, so this is not
      external validation, and a high score here may not carry over to new news.
    - Group-aware splitting does not guarantee independence by publisher,
      topic or time.
    - MinHash LSH is approximate and can miss near-duplicates.
    - The source-cue list was picked by hand, and the F1 gap it produces is not
      proof of leakage.
    - Dataset labels and source mix can carry bias.
    - SVM decision scores are not calibrated probabilities.
    - Feature contributions show associations learned by the model, not causes.
    - The model never checks claims against live sources.
    """)
    st.caption(
        "The saved model is trained on the full cleaned dataset for the demo, "
        "so its performance should not be quoted as independent test results."
    )
