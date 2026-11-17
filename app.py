import pickle
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features_extraction import MODEL_FEATURE_NAMES, Extractor

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
DATASET_PATH = ROOT / "dataset_2.csv"
MODEL_COLOR = "#8b9cff"
PHISHING_COLOR = "#ff6b81"
LEGITIMATE_COLOR = "#36d6b5"
FEATURE_LABELS = {
    "PathLength": "Path length",
    "UrlLength": "Total URL length",
    "HostnameLength": "Hostname length",
    "NumNumericChars": "Numeric characters",
    "PathLevel": "Path depth",
    "NumDots": "Dots",
    "SubdomainLevel": "Hostname labels",
    "NumUnderscore": "Underscores",
    "NumDashInHostname": "Host/path dashes",
    "RandomString": "Random-looking token",
    "NumDash": "Dashes",
    "QueryLength": "Query length",
    "NumQueryComponents": "Query parameters",
    "IpAddress": "IP address host",
    "TinyURL": "Shortener pattern",
}
SHOWCASE_METRICS = {
    "Random Forest accuracy": "81.25%",
    "Stacking + Extra Trees accuracy": "82.00%",
    "Ensemble precision": "80.70%",
    "Ensemble recall": "81.90%",
}
extractor = Extractor()

APP_STYLES = """
<style>
:root {
  --ink: #e8edf7;
  --muted: #9aa8bf;
  --line: rgba(165, 184, 218, 0.14);
  --panel: rgba(17, 29, 49, 0.82);
  --accent: #8b9cff;
  --mint: #36d6b5;
}

.stApp {
  background:
    radial-gradient(ellipse at 82% 0%, rgba(86, 102, 213, 0.18), transparent 34rem),
    radial-gradient(ellipse at 6% 30%, rgba(25, 167, 154, 0.10), transparent 32rem),
    #080e19;
  color: var(--ink);
  font-family: 'Segoe UI', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1728 0%, #0a111e 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
.block-container {
  max-width: 1380px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}

h1, h2, h3 {
  color: #f3f6fc;
  font-family: 'Segoe UI', sans-serif;
  letter-spacing: -0.035em;
}

.brand-mark {
  color: var(--accent);
  font-family: 'Segoe UI', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.brand-caption, .muted-copy {
  color: var(--muted);
  font-size: 0.9rem;
}
.hero {
  background: linear-gradient(120deg, rgba(23, 38, 64, 0.92), rgba(15, 25, 43, 0.78));
  border: 1px solid var(--line);
  border-radius: 24px;
  margin: 0.4rem 0 1.6rem;
  padding: 2rem 2.2rem;
  position: relative;
  overflow: hidden;
}
.hero:after {
  background: radial-gradient(circle, rgba(139, 156, 255, 0.24), transparent 68%);
  content: "";
  height: 22rem;
  position: absolute;
  right: -7rem;
  top: -13rem;
  width: 22rem;
}
.eyebrow {
  color: #aab8ff;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}
.hero h1 {
  font-size: clamp(2rem, 4vw, 3.25rem);
  line-height: 1.08;
  margin: 0.7rem 0 0.6rem;
}
.hero p {
  color: #aebbd0;
  font-size: 1.03rem;
  margin: 0;
  max-width: 48rem;
}

[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(22, 36, 59, 0.92), rgba(14, 24, 41, 0.9));
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem 1.1rem;
}
[data-testid="stMetricLabel"] { color: #aab8ce; }
[data-testid="stMetricValue"] {
  color: #f4f7ff;
  font-family: 'Segoe UI', sans-serif;
}

.stButton > button, [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(110deg, #8797ff, #6678ee);
  border: 0;
  border-radius: 11px;
  color: #0c1220;
  font-weight: 700;
  min-height: 2.85rem;
  transition: filter 120ms ease, transform 120ms ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  filter: brightness(1.09);
  transform: translateY(-1px);
}

[data-baseweb="tab-list"] { gap: 0.5rem; }
[data-baseweb="tab"] {
  background: rgba(17, 29, 49, 0.55);
  border: 1px solid transparent;
  border-radius: 10px 10px 0 0;
  color: #9eacc2;
  padding: 0.75rem 1rem;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: rgba(32, 45, 70, 0.9);
  border-color: var(--line);
  color: #eef1ff;
}

[data-testid="stTextInput"] input {
  background: rgba(7, 14, 26, 0.78);
  border-color: rgba(156, 175, 211, 0.25);
  border-radius: 10px;
  color: #f3f6fc;
  min-height: 2.85rem;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}
[data-testid="stExpander"] {
  background: rgba(17, 29, 49, 0.5);
  border: 1px solid var(--line);
  border-radius: 14px;
}
hr { border-color: var(--line); }
</style>
"""


@st.cache_resource
def load_model(model_path: str, modified_time: float) -> Pipeline:
    path = Path(model_path)
    if path.stat().st_mtime != modified_time:
        raise RuntimeError("The model changed while loading. Please retry.")

    with path.open("rb") as model_file:
        model = pickle.load(model_file)

    if not isinstance(model, Pipeline):
        raise TypeError("model.pkl must contain the trained preprocessing pipeline.")
    if not isinstance(model.named_steps.get("classifier"), RandomForestClassifier):
        raise TypeError("model.pkl must contain the selected Random Forest classifier.")
    if tuple(getattr(model, "feature_names_in_", ())) != MODEL_FEATURE_NAMES:
        raise ValueError(
            "model.pkl feature schema does not match the app. "
            "Run `python train_model.py` to rebuild it."
        )
    return model


@st.cache_data
def load_holdout_analytics(model_modified_time: float) -> dict:
    data = pd.read_csv(DATASET_PATH)
    features = data.loc[:, MODEL_FEATURE_NAMES]
    labels = data["Label"]
    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )
    model = load_model(str(MODEL_PATH), model_modified_time)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    class_counts = labels.value_counts().reindex([0, 1], fill_value=0)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "class_counts": [int(class_counts.loc[0]), int(class_counts.loc[1])],
        "test_size": int(len(y_test)),
    }


def normalize_url(value: str) -> str:
    url = value.strip()
    if not url:
        raise ValueError("Enter a URL to analyze.")
    if "://" not in url:
        url = f"https://{url}"

    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        _ = parsed_url.port
    except ValueError as error:
        raise ValueError("Enter a valid HTTP or HTTPS URL.") from error

    if (
        parsed_url.scheme.lower() not in {"http", "https"}
        or not hostname
        or any(character.isspace() for character in parsed_url.netloc)
    ):
        raise ValueError("Enter a valid HTTP or HTTPS URL.")

    return url


def score_gauge(score: float) -> go.Figure:
    color = PHISHING_COLOR if score >= 0.5 else LEGITIMATE_COLOR
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#f4f7ff"}},
            title={
                "text": "Phishing score<br><span style='font-size:12px;color:#9aa8bf'>uncalibrated model estimate</span>",
                "font": {"size": 15, "color": "#cbd5e1"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 50, 100],
                    "tickcolor": "#8290a8",
                    "tickfont": {"color": "#9aa8bf"},
                },
                "bar": {"color": color, "thickness": 0.24},
                "bgcolor": "rgba(10, 17, 30, 0.8)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(54, 214, 181, 0.13)"},
                    {"range": [50, 100], "color": "rgba(255, 107, 129, 0.13)"},
                ],
                "threshold": {
                    "line": {"color": "#f5c46b", "width": 3},
                    "value": 50,
                    "thickness": 0.78,
                },
            },
        )
    )
    figure.update_layout(
        height=250,
        margin={"l": 22, "r": 22, "t": 58, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8edf7"},
    )
    return figure


def feature_importance_chart(model: Pipeline) -> go.Figure:
    classifier = model.named_steps["classifier"]
    ranked = sorted(
        zip(MODEL_FEATURE_NAMES, classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    names = [FEATURE_LABELS[name] for name, _ in reversed(ranked)]
    values = [float(value) for _, value in reversed(ranked)]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=MODEL_COLOR,
            hovertemplate="%{y}<br>Relative importance: %{x:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=340,
        margin={"l": 10, "r": 18, "t": 18, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#aebbd0"},
        xaxis={"title": "Relative importance", "gridcolor": "rgba(165,184,218,0.12)"},
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    return figure


def confusion_chart(matrix: list[list[int]]) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Legitimate", "Phishing"],
            y=["Legitimate", "Phishing"],
            colorscale=[
                [0, "#111c30"],
                [0.5, "#4e65bd"],
                [1, "#36d6b5"],
            ],
            showscale=False,
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 12, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
    )
    return figure


def class_balance_chart(class_counts: list[int]) -> go.Figure:
    figure = go.Figure(
        go.Pie(
            labels=["Legitimate", "Phishing"],
            values=class_counts,
            hole=0.68,
            sort=False,
            marker={"colors": [LEGITIMATE_COLOR, PHISHING_COLOR]},
            textinfo="label+percent",
            textfont={"color": "#e8edf7", "size": 12},
            hovertemplate="%{label}: %{value:,} rows<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 12, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            {
                "text": "10K<br><span style='font-size:11px'>URLs</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 22, "color": "#f4f7ff"},
            }
        ],
    )
    return figure


def render_result(result: dict, model: Pipeline) -> None:
    is_phishing = result["prediction"] == 1
    result_color = PHISHING_COLOR if is_phishing else LEGITIMATE_COLOR
    verdict = "Likely phishing" if is_phishing else "Likely legitimate"

    st.markdown("### Analysis result")
    summary, gauge = st.columns([1.05, 1.35], gap="large")
    with summary:
        with st.container(border=True):
            st.markdown(
                f"<div class='eyebrow'>MODEL VERDICT</div>"
                f"<h2 style='color:{result_color};margin:.5rem 0 0'>{verdict}</h2>",
                unsafe_allow_html=True,
            )
            st.caption("Analyzed host")
            st.code(result["host"], language="text")
            st.caption("Model output is a screening signal, not a safety guarantee.")
            if is_phishing:
                st.warning(
                    "Avoid entering credentials or payment details. Verify the "
                    "domain through a trusted channel."
                )
            else:
                st.info(
                    "Still check the spelling and destination before sharing "
                    "sensitive information."
                )
    with gauge:
        st.plotly_chart(
            score_gauge(result["score"]),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.markdown("#### Random Forest feature importance")
    st.caption(
        "Overall feature importance from the fitted model; this is not a "
        "per-URL explanation or evidence of causation."
    )
    st.plotly_chart(
        feature_importance_chart(model),
        width="stretch",
        config={"displayModeBar": False},
    )

    with st.expander("Inspect the 15 model inputs"):
        feature_rows = [
            {"Feature": FEATURE_LABELS[name], "Value": value}
            for name, value in result["features"].items()
        ]
        st.dataframe(
            pd.DataFrame(feature_rows),
            hide_index=True,
            width="stretch",
        )


def render_scan_tab() -> None:
    st.markdown("### Quick scan")
    with st.form("url_scan", clear_on_submit=False):
        url_column, button_column = st.columns([4, 2], vertical_alignment="bottom")
        with url_column:
            input_url = st.text_input(
                "Website URL",
                placeholder="https://example.com/login",
                label_visibility="collapsed",
            )
        with button_column:
            submitted = st.form_submit_button(
                "Analyze URL",
                type="primary",
                use_container_width=True,
            )

    st.caption(
        "No request is sent to the submitted website. The app analyzes URL "
        "structure only."
    )
    if submitted:
        try:
            url = normalize_url(input_url)
        except ValueError as error:
            st.session_state.pop("last_result", None)
            st.error(str(error))
        else:
            model_path = str(MODEL_PATH)
            model = load_model(model_path, MODEL_PATH.stat().st_mtime)
            features = extractor.extract_model_features(url)
            feature_frame = pd.DataFrame(
                [features],
                columns=MODEL_FEATURE_NAMES,
            )
            classes = list(model.classes_)
            phishing_index = classes.index(1)
            score = float(model.predict_proba(feature_frame)[0][phishing_index])
            prediction = int(model.predict(feature_frame)[0])
            host = urlsplit(url).hostname or url
            result = {
                "host": host,
                "features": features,
                "prediction": prediction,
                "score": score,
            }
            st.session_state["last_result"] = result
            history = st.session_state.setdefault("analysis_history", [])
            history.insert(
                0,
                {
                    "Time": datetime.now().astimezone().strftime("%H:%M:%S"),
                    "Host": host,
                    "Verdict": "Phishing" if prediction == 1 else "Legitimate",
                    "Score": round(score * 100, 1),
                },
            )
            del history[50:]

    result = st.session_state.get("last_result")
    if result is None:
        st.markdown(
            "<div class='muted-copy'>"
            "Enter a URL above to see a model score, feature-level signals, "
            "and an explanation of the result."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        render_result(result, model)


def render_session_analytics() -> None:
    history = st.session_state.get("analysis_history", [])
    st.markdown("### Your session at a glance")
    if not history:
        st.info("Run a URL analysis to populate these live session charts.")
        return

    history_frame = pd.DataFrame(history)
    flagged = int((history_frame["Verdict"] == "Phishing").sum())
    legitimate = len(history_frame) - flagged
    average_score = float(history_frame["Score"].mean())

    total_metric, flagged_metric, score_metric = st.columns(3)
    total_metric.metric("URLs analyzed", len(history_frame))
    flagged_metric.metric("Flagged by model", flagged)
    score_metric.metric("Avg. phishing score", f"{average_score:.1f}%")

    chart_column, history_column = st.columns([1, 1.35], gap="large")
    with chart_column:
        st.markdown("#### Verdict distribution")
        donut = go.Figure(
            go.Pie(
                labels=["Phishing", "Legitimate"],
                values=[flagged, legitimate],
                hole=0.68,
                sort=False,
                marker={"colors": [PHISHING_COLOR, LEGITIMATE_COLOR]},
                textinfo="label+value",
                textfont={"color": "#e8edf7", "size": 12},
                hovertemplate="%{label}: %{value} scans<extra></extra>",
            )
        )
        donut.update_layout(
            height=300,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[
                {
                    "text": f"{len(history_frame)}<br><span style='font-size:11px'>scans</span>",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 22, "color": "#f4f7ff"},
                }
            ],
        )
        st.plotly_chart(donut, width="stretch", config={"displayModeBar": False})

    with history_column:
        st.markdown("#### Recent scans")
        st.dataframe(
            history_frame,
            hide_index=True,
            width="stretch",
            height=300,
        )

    st.caption("History is stored in this browser session only; full URLs are not retained.")
    if st.button("Clear session history"):
        st.session_state["analysis_history"] = []
        st.session_state.pop("last_result", None)
        st.rerun()


def render_model_insights() -> None:
    st.markdown("### Showcase results")
    st.caption(
        "Static figures from the supplied research summary. The combined "
        "Stacking + Extra Trees scores are reported separately from the "
        "Random Forest used for live URL scans."
    )
    first_row = st.columns(2)
    second_row = st.columns(2)
    for column, (label, value) in zip(
        (*first_row, *second_row),
        SHOWCASE_METRICS.items(),
    ):
        column.metric(label, value)

    st.divider()
    analytics = load_holdout_analytics(MODEL_PATH.stat().st_mtime)
    st.markdown("### Random Forest holdout performance")
    st.caption(
        "Calculated from the checked-in Random Forest pipeline on a fixed "
        "holdout split; separate from the static showcase figures above."
    )
    accuracy, precision = st.columns(2)
    accuracy.metric("Accuracy", f"{analytics['accuracy']:.2%}")
    precision.metric("Precision", f"{analytics['precision']:.2%}")
    recall, f1 = st.columns(2)
    recall.metric("Recall", f"{analytics['recall']:.2%}")
    f1.metric("F1 score", f"{analytics['f1']:.2%}")

    matrix_column, balance_column = st.columns([1.2, 0.8], gap="large")
    with matrix_column:
        st.markdown("#### Holdout confusion matrix")
        st.plotly_chart(
            confusion_chart(analytics["confusion_matrix"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            f"Test split: {analytics['test_size']:,} URLs. "
            "Rows are actual labels; columns are model predictions."
        )
    with balance_column:
        st.markdown("#### Training data balance")
        st.plotly_chart(
            class_balance_chart(analytics["class_counts"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption("The source training table contains 10,000 labeled URLs.")


def main() -> None:
    st.set_page_config(
        page_title="PhishProbe | URL Intelligence",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.session_state.setdefault("analysis_history", [])

    with st.sidebar:
        st.markdown("<div class='brand-mark'>PHISHPROBE</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='brand-caption'>Machine-learning URL intelligence</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**DETECTOR**")
        st.markdown("Random Forest")
        st.markdown("15 URL features")
        st.markdown("MinMax-scaled pipeline")
        st.divider()
        st.caption(
            "Research demo only. A legitimate prediction does not guarantee a "
            "safe website."
        )

    st.markdown(
        "<div class='hero'>"
        "<div class='eyebrow'>PHISHING URL INTELLIGENCE</div>"
        "<h1>Know the link<br>before you trust it.</h1>"
        "<p>Inspect URL structure with a transparent machine-learning demo. "
        "Get a fast verdict, model score, and the signals behind it.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    scan_tab, session_tab, model_tab = st.tabs(
        ["URL scanner", "Session analytics", "Model insights"]
    )
    with scan_tab:
        render_scan_tab()
    with session_tab:
        render_session_analytics()
    with model_tab:
        render_model_insights()


if __name__ == "__main__":
    main()
