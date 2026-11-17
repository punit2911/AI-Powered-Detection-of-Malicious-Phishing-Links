import pickle
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from features_extraction import MODEL_FEATURE_NAMES, Extractor

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")
extractor = Extractor()


def load_model() -> Pipeline:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    if not isinstance(model, Pipeline):
        raise TypeError("model.pkl must contain the trained preprocessing pipeline.")
    if tuple(getattr(model, "feature_names_in_", ())) != MODEL_FEATURE_NAMES:
        raise ValueError(
            "model.pkl feature schema does not match the app. "
            "Run `python train_model.py` to rebuild it."
        )
    return model


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


def main() -> None:
    st.set_page_config(page_title="Phishing URL Detector")
    st.title("Phishing URL Detector")
    st.write(
        "Enter a URL to estimate whether its structure resembles the phishing "
        "examples used to train this model."
    )
    input_url = st.text_input("Website URL", placeholder="https://example.com")

    if not st.button("Analyze"):
        return

    try:
        url = normalize_url(input_url)
    except ValueError as error:
        st.error(str(error))
        return

    features = extractor.extract_model_features(url)
    feature_frame = pd.DataFrame(
        [features],
        columns=MODEL_FEATURE_NAMES,
    )
    model = load_model()
    prediction = int(model.predict(feature_frame)[0])

    if prediction == 1:
        st.subheader("Likely a phishing URL")
        st.image(str(MODEL_PATH.with_name("image_red_cross.png")), width=40)
        st.warning(
            "Do not enter passwords or payment details. Verify the domain "
            "through a trusted source."
        )
    else:
        st.subheader("Likely a legitimate URL")
        st.image(str(MODEL_PATH.with_name("image_green_tick.png")), width=40)
        st.info(
            "A legitimate prediction is not a safety guarantee. Check the "
            "domain carefully before sharing sensitive information."
        )

    st.caption(
        "Experimental machine-learning result only; this classifier can make "
        "mistakes and is not a substitute for security tools."
    )
    st.subheader("Model input features")
    for name, value in features.items():
        st.text(f"{name}: {value}")


if __name__ == "__main__":
    main()
