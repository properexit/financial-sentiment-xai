import os
import sys
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.explain import GradientInputExplainer


# Streamlit configuration
st.set_page_config(
    page_title="Financial Sentiment XAI",
    layout="centered"
)

st.title("📊 Financial Sentiment Analysis (Explainable)")
st.write(
    "Enter a short financial sentence. "
    "The model predicts sentiment and highlights which words influenced the decision."
)


@st.cache_resource
def load_explainer():
    """
    Load the explainer once and cache it.

    Loading the tokenizer and model is expensive, so we avoid
    reinitializing them on every interaction.
    """
    return GradientInputExplainer()


explainer = load_explainer()

text = st.text_area(
    "Input sentence",
    height=100,
    placeholder="e.g. The company reported strong profit growth this quarter."
)

if st.button("Analyze") and text.strip():
    result = explainer.explain(text)

    prediction = result["prediction"]
    tokens = result["tokens"]
    scores = np.array(result["scores"])

    st.subheader("Prediction")
    st.success(prediction.capitalize())

    st.subheader("Token-level explanation")

    # Normalize scores for consistent visual scaling
    scores = scores / (scores.max() + 1e-8)

    def render_token(token: str, score: float) -> str:
        """
        Render a single token with background intensity proportional
        to its attribution score.
        """
        alpha = min(0.8, score)
        return (
            f"<span style='background-color: rgba(255, 165, 0, {alpha}); "
            f"padding: 2px 4px; margin: 1px; border-radius: 4px'>"
            f"{token}</span>"
        )

    # Skip special tokens used internally by the model
    html_tokens = [
        render_token(tok, score)
        for tok, score in zip(tokens, scores)
        if tok not in {"[CLS]", "[SEP]"}
    ]

    st.markdown(
        "<div style='line-height: 2.2'>" + " ".join(html_tokens) + "</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Darker highlights indicate higher influence on the model's prediction "
        "(Gradient × Input attribution)."
    )