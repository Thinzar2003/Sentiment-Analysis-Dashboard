"""
Sentiment Analysis Dashboard — Streamlit App
Loads model directly from HuggingFace Hub — no FastAPI needed.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from transformers import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .positive { color: #4ade80; font-size: 1.4rem; font-weight: 700; }
    .negative { color: #f87171; font-size: 1.4rem; font-weight: 700; }
    .neutral  { color: #94a3b8; font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="Thinzar2003/sentiment-distilbert",
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ About")
    st.markdown(
        "Fine-tuned **DistilBERT** model for binary sentiment classification "
        "(Positive / Negative) on IMDB movie reviews."
    )
    st.markdown("**Model**: `Thinzar2003/sentiment-distilbert`")
    st.markdown("**Dataset**: IMDB (50k reviews)")
    st.markdown("**Built with**: HuggingFace + Streamlit")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💬 Sentiment Analysis Dashboard")
st.caption("Powered by fine-tuned DistilBERT · Built with HuggingFace + Streamlit")
st.divider()

# ── Load model with spinner ───────────────────────────────────────────────────
with st.spinner("Loading model from HuggingFace Hub... (first load takes ~30s)"):
    classifier = load_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔤 Single Text", "📂 Batch Analysis (CSV)"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Text
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    examples = [
        "This movie was absolutely fantastic! A masterpiece.",
        "Terrible film. Waste of 2 hours, extremely boring.",
        "It was alright, had some good moments but nothing special.",
    ]

    st.markdown("**Try these examples:**")
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex[:40] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["input_text"] = ex

    user_text = st.text_area(
        "Or type your own text below:",
        value=st.session_state.get("input_text", ""),
        placeholder="Type a review, tweet, or any text here...",
        height=150,
        key="input_text",
    )

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                t0 = time.time()
                result = classifier(user_text)[0]
                latency = round((time.time() - t0) * 1000, 1)

            st.divider()
            r1, r2, r3 = st.columns(3)

            label = result["label"]
            score = result["score"]
            emoji = "😊" if label == "POSITIVE" else "😞"
            color_cls = "positive" if label == "POSITIVE" else "negative"

            with r1:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='color:#94a3b8;font-size:.85rem'>Sentiment</div>"
                    f"<div class='{color_cls}'>{emoji} {label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='color:#94a3b8;font-size:.85rem'>Confidence</div>"
                    f"<div class='positive'>{round(score*100,1)}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='color:#94a3b8;font-size:.85rem'>Latency</div>"
                    f"<div class='neutral'>{latency} ms</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            fig = go.Figure(go.Bar(
                x=[score, 1 - score],
                y=["Predicted", "Other"],
                orientation="h",
                marker_color=["#4ade80" if label == "POSITIVE" else "#f87171", "#2d3250"],
                text=[f"{score*100:.1f}%", f"{(1-score)*100:.1f}%"],
                textposition="inside",
            ))
            fig.update_layout(
                title="Confidence Scores",
                paper_bgcolor="#1a1d2e",
                plot_bgcolor="#1a1d2e",
                font_color="#e2e8f0",
                height=180,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch CSV Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a CSV file with a **`text`** column (max 100 rows).")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    with col_b:
        sample = pd.DataFrame({"text": [
            "Amazing product, highly recommend!",
            "Worst experience ever, do not buy.",
            "Pretty good for the price.",
            "Completely broke after one use.",
            "Exceeded my expectations, love it!",
        ]})
        st.download_button(
            "📥 Download sample CSV",
            data=sample.to_csv(index=False),
            file_name="sample_reviews.csv",
            mime="text/csv",
        )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            st.info(f"Loaded **{len(df)}** rows. Previewing first 5:")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Run Batch Analysis", type="primary"):
                texts = df["text"].astype(str).tolist()[:100]

                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    raw = classifier(texts)

                results_df = pd.DataFrame([
                    {
                        "text": text,
                        "label": r["label"],
                        "confidence": round(r["score"] * 100, 1),
                    }
                    for text, r in zip(texts, raw)
                ])

                st.divider()
                st.subheader("📊 Summary")
                pos = sum(1 for r in raw if r["label"] == "POSITIVE")
                neg = len(raw) - pos

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Analyzed", len(raw))
                m2.metric("Positive 😊", pos)
                m3.metric("Negative 😞", neg)
                m4.metric("Avg Confidence", f"{results_df['confidence'].mean():.1f}%")

                ch1, ch2 = st.columns(2)
                with ch1:
                    pie = px.pie(
                        names=["POSITIVE", "NEGATIVE"],
                        values=[pos, neg],
                        color=["POSITIVE", "NEGATIVE"],
                        color_discrete_map={"POSITIVE": "#4ade80", "NEGATIVE": "#f87171"},
                        title="Sentiment Distribution",
                    )
                    pie.update_layout(paper_bgcolor="#1a1d2e", font_color="#e2e8f0")
                    st.plotly_chart(pie, use_container_width=True)

                with ch2:
                    hist = px.histogram(
                        results_df, x="confidence", color="label",
                        color_discrete_map={"POSITIVE": "#4ade80", "NEGATIVE": "#f87171"},
                        title="Confidence Distribution", nbins=20,
                    )
                    hist.update_layout(
                        paper_bgcolor="#1a1d2e",
                        plot_bgcolor="#1a1d2e",
                        font_color="#e2e8f0",
                    )
                    st.plotly_chart(hist, use_container_width=True)

                st.subheader("📋 Results Table")
                st.dataframe(
                    results_df.style.applymap(
                        lambda v: "color: #4ade80" if v == "POSITIVE" else "color: #f87171",
                        subset=["label"],
                    ),
                    use_container_width=True,
                    height=300,
                )

                st.download_button(
                    "📥 Download Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )
