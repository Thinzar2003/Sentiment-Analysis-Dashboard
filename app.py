"""
Sentiment Analysis Dashboard — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import time

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
    .main { background: #0f1117; }
    .stApp { background: #0f1117; }
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
    .score-bar { height: 8px; border-radius: 4px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="URL of your running FastAPI backend",
    )
    st.divider()
    st.markdown("### About")
    st.markdown(
        "Fine-tuned **DistilBERT** model for binary sentiment classification "
        "(Positive / Negative) on IMDB movie reviews."
    )
    st.markdown("**Model**: `distilbert-base-uncased`")
    st.markdown("**Dataset**: IMDB (50k reviews)")

    # Health check
    st.divider()
    if st.button("🔍 Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.ok:
                st.success("API is online ✅")
            else:
                st.error("API returned an error")
        except Exception:
            st.error("Cannot connect to API")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💬 Sentiment Analysis Dashboard")
st.caption("Powered by fine-tuned DistilBERT · Built with HuggingFace + FastAPI + Streamlit")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔤 Single Text", "📂 Batch Analysis (CSV)"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Text
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        user_text = st.text_area(
            "Enter text to analyze",
            placeholder="Type a review, tweet, or any text here...",
            height=150,
        )

    with col2:
        st.markdown("**Try these examples:**")
        examples = [
            "This movie was absolutely fantastic! A masterpiece.",
            "Terrible film. Waste of 2 hours, extremely boring.",
            "It was alright, had some good moments but nothing special.",
        ]
        for ex in examples:
            if st.button(ex[:45] + "...", key=ex):
                user_text = ex

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    resp = requests.post(
                        f"{api_url}/predict",
                        json={"text": user_text},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    # Display result
                    st.divider()
                    r1, r2, r3 = st.columns(3)

                    with r1:
                        label = result["label"]
                        color_cls = "positive" if label == "POSITIVE" else "negative"
                        emoji = "😊" if label == "POSITIVE" else "😞"
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div style='color:#94a3b8;font-size:.85rem'>Sentiment</div>"
                            f"<div class='{color_cls}'>{emoji} {label}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    with r2:
                        score_pct = round(result["score"] * 100, 1)
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div style='color:#94a3b8;font-size:.85rem'>Confidence</div>"
                            f"<div class='positive'>{score_pct}%</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    with r3:
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div style='color:#94a3b8;font-size:.85rem'>Latency</div>"
                            f"<div class='neutral'>{result['latency_ms']} ms</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Confidence bar chart
                    fig = go.Figure(go.Bar(
                        x=[result["score"], 1 - result["score"]],
                        y=["Predicted", "Other"],
                        orientation="h",
                        marker_color=["#4ade80" if label == "POSITIVE" else "#f87171", "#2d3250"],
                        text=[f"{result['score']*100:.1f}%", f"{(1-result['score'])*100:.1f}%"],
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

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API. Make sure your FastAPI server is running.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch CSV Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a CSV file with a **`text`** column (max 100 rows per batch).")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    with col_b:
        # Sample CSV download
        sample = pd.DataFrame({
            "text": [
                "Amazing product, highly recommend!",
                "Worst experience ever, do not buy.",
                "Pretty good for the price.",
                "Completely broke after one use.",
                "Exceeded my expectations, love it!",
            ]
        })
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

                progress = st.progress(0, text="Sending to API...")
                try:
                    resp = requests.post(
                        f"{api_url}/predict/batch",
                        json={"texts": texts},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    progress.progress(100, text="Done!")
                    time.sleep(0.3)
                    progress.empty()

                    results_df = pd.DataFrame([
                        {"text": r["text"], "label": r["label"], "confidence": round(r["score"] * 100, 1)}
                        for r in data["results"]
                    ])
                    summary = data["summary"]

                    # ── Summary metrics ────────────────────────────────────
                    st.divider()
                    st.subheader("📊 Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Analyzed", summary["total"])
                    m2.metric("Positive 😊", summary["positive"])
                    m3.metric("Negative 😞", summary["negative"])
                    m4.metric("Avg Confidence", f"{summary['avg_confidence']*100:.1f}%")

                    # ── Charts ─────────────────────────────────────────────
                    ch1, ch2 = st.columns(2)

                    with ch1:
                        pie = px.pie(
                            names=["POSITIVE", "NEGATIVE"],
                            values=[summary["positive"], summary["negative"]],
                            color=["POSITIVE", "NEGATIVE"],
                            color_discrete_map={"POSITIVE": "#4ade80", "NEGATIVE": "#f87171"},
                            title="Sentiment Distribution",
                        )
                        pie.update_layout(
                            paper_bgcolor="#1a1d2e",
                            font_color="#e2e8f0",
                        )
                        st.plotly_chart(pie, use_container_width=True)

                    with ch2:
                        hist = px.histogram(
                            results_df,
                            x="confidence",
                            color="label",
                            color_discrete_map={"POSITIVE": "#4ade80", "NEGATIVE": "#f87171"},
                            title="Confidence Distribution",
                            nbins=20,
                        )
                        hist.update_layout(
                            paper_bgcolor="#1a1d2e",
                            plot_bgcolor="#1a1d2e",
                            font_color="#e2e8f0",
                        )
                        st.plotly_chart(hist, use_container_width=True)

                    # ── Results table ──────────────────────────────────────
                    st.subheader("📋 Results Table")
                    st.dataframe(
                        results_df.style.applymap(
                            lambda v: "color: #4ade80" if v == "POSITIVE" else "color: #f87171",
                            subset=["label"],
                        ),
                        use_container_width=True,
                        height=300,
                    )

                    # Download results
                    st.download_button(
                        "📥 Download Results CSV",
                        data=results_df.to_csv(index=False),
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API.")
                except Exception as e:
                    st.error(f"Error: {e}")
