"""
Sentiment Analysis Dashboard — Redesigned with Purple/Gradient Theme
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
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(236,72,153,0.1));
    border-radius: 20px;
    border: 1px solid rgba(139,92,246,0.3);
    margin-bottom: 2rem;
    animation: fadeIn 0.8s ease;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

.result-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(236,72,153,0.08));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    animation: slideUp 0.5s ease;
}
.result-card .label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-card .value { font-size: 1.8rem; font-weight: 700; }

.positive { color: #4ade80; }
.negative { color: #f87171; }
.neutral  { color: #a78bfa; }

.chip {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #c4b5fd;
    margin: 2px;
}

.stButton > button {
    background: linear-gradient(135deg, rgba(139,92,246,0.3), rgba(236,72,153,0.2)) !important;
    border: 1px solid rgba(139,92,246,0.5) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(139,92,246,0.5), rgba(236,72,153,0.4)) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(139,92,246,0.3) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(124,58,237,0.6) !important;
    transform: translateY(-2px) !important;
}

.stTextArea textarea {
    background: rgba(15,12,41,0.8) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533, #0f0c29) !important;
    border-right: 1px solid rgba(139,92,246,0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(139,92,246,0.1) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
}

hr { border-color: rgba(139,92,246,0.3) !important; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PAPER_BG = "rgba(26,5,51,0.6)"
FONT_CLR = "#e2e8f0"
PURPLE   = "#a78bfa"
GREEN    = "#4ade80"
RED      = "#f87171"

def plot_layout(title="", h=280):
    return dict(
        title=dict(text=title, font=dict(color=PURPLE, size=14)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_CLR, family="Inter"),
        height=h,
        margin=dict(l=20, r=20, t=45, b=20),
    )

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Thinzar2003/sentiment-distilbert")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:2.5rem'>🧠</div>
        <div style='font-size:1.2rem; font-weight:700;
             background:linear-gradient(135deg,#a78bfa,#ec4899);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            SentimentIQ
        </div>
        <div style='color:#64748b; font-size:0.75rem;'>NLP Portfolio Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🔬 Model Info")
    st.markdown("""
    <div style='font-size:0.85rem; color:#94a3b8; line-height:2'>
    <span class='chip'>DistilBERT</span>
    <span class='chip'>HuggingFace</span><br><br>
    📦 <b style='color:#c4b5fd'>Model</b><br>
    <code style='font-size:0.75rem'>Thinzar2003/sentiment-distilbert</code><br><br>
    📚 <b style='color:#c4b5fd'>Dataset</b><br>IMDB (50k reviews)<br><br>
    🎯 <b style='color:#c4b5fd'>Task</b><br>Binary Sentiment Classification<br><br>
    ⚡ <b style='color:#c4b5fd'>Stack</b><br>PyTorch · Streamlit · HuggingFace
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Model Metrics")
    perf = go.Figure(go.Bar(
        x=[93, 92, 93],
        y=["Accuracy", "F1 Score", "Precision"],
        orientation="h",
        marker=dict(color=["#7c3aed", "#db2777", "#a78bfa"]),
        text=["93%", "92%", "93%"],
        textposition="inside",
        insidetextfont=dict(color="white", size=11),
    ))
    perf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_CLR, size=11),
        height=150,
        margin=dict(l=0, r=10, t=5, b=0),
        xaxis=dict(range=[0,100], visible=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(perf, use_container_width=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🧠 SentimentIQ</h1>
    <p>Real-time sentiment analysis powered by fine-tuned DistilBERT · HuggingFace + Streamlit</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("✨ Loading AI model..."):
    classifier = load_model()

tab1, tab2 = st.tabs(["🔤  Single Text Analysis", "📂  Batch CSV Analysis"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    examples = [
        "This movie was absolutely fantastic! A total masterpiece.",
        "Terrible film. Complete waste of 2 hours, extremely boring.",
        "It was alright, had some good moments but nothing special.",
    ]

    st.markdown("##### ⚡ Quick examples")
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex[:42] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["input_text"] = ex

    st.markdown("")
    user_text = st.text_area(
        "✍️ Or type your own text:",
        value=st.session_state.get("input_text", ""),
        placeholder="Paste a review, tweet, customer feedback, or any text...",
        height=130,
        key="input_text",
    )

    st.markdown("")
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🧠 Analyzing..."):
                t0 = time.time()
                result = classifier(user_text)[0]
                latency = round((time.time() - t0) * 1000, 1)

            label     = result["label"]
            score     = result["score"]
            emoji     = "😊" if label == "POSITIVE" else "😞"
            color_cls = "positive" if label == "POSITIVE" else "negative"
            bar_color = GREEN if label == "POSITIVE" else RED

            st.divider()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class='result-card'>
                    <div class='label'>Sentiment</div>
                    <div class='value {color_cls}'>{emoji} {label}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='result-card'>
                    <div class='label'>Confidence</div>
                    <div class='value neutral'>{round(score*100,1)}%</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class='result-card'>
                    <div class='label'>Latency</div>
                    <div class='value neutral'>⚡ {latency} ms</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            ch1, ch2 = st.columns(2)

            with ch1:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(score * 100, 1),
                    number={"suffix": "%", "font": {"color": bar_color, "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": FONT_CLR},
                        "bar": {"color": bar_color, "thickness": 0.25},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "bordercolor": "rgba(139,92,246,0.3)",
                        "steps": [
                            {"range": [0,  50], "color": "rgba(248,113,113,0.15)"},
                            {"range": [50, 80], "color": "rgba(167,139,250,0.15)"},
                            {"range": [80,100], "color": "rgba(74,222,128,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": PURPLE, "width": 3},
                            "thickness": 0.75,
                            "value": 80,
                        },
                    },
                    title={"text": "Confidence Gauge", "font": {"color": PURPLE, "size": 13}},
                ))
                gauge.update_layout(**plot_layout(h=260))
                st.plotly_chart(gauge, use_container_width=True)

            with ch2:
                bar = go.Figure(go.Bar(
                    x=["POSITIVE", "NEGATIVE"],
                    y=[score if label=="POSITIVE" else 1-score,
                       score if label=="NEGATIVE" else 1-score],
                    marker=dict(color=[GREEN, RED], opacity=0.85),
                    text=[f"{(score if label=='POSITIVE' else 1-score)*100:.1f}%",
                          f"{(score if label=='NEGATIVE' else 1-score)*100:.1f}%"],
                    textposition="outside",
                    textfont=dict(color=FONT_CLR),
                ))
                bar.update_layout(
                    **plot_layout("Score Distribution", h=260),
                    yaxis=dict(range=[0, 1.2], showgrid=False, visible=False),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(bar, use_container_width=True)

            st.markdown(f"""
            <div style='background:rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.3);
                        border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;'>
                <div style='color:#94a3b8; font-size:0.75rem; margin-bottom:0.3rem'>ANALYZED TEXT</div>
                <div style='color:#e2e8f0; font-size:0.95rem; line-height:1.6'>"{user_text}"</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a CSV with a **`text`** column (max 100 rows).")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        uploaded_file = st.file_uploader("📎 Upload CSV", type=["csv"])
    with col_b:
        sample = pd.DataFrame({"text": [
            "Amazing product, highly recommend!",
            "Worst experience ever, do not buy.",
            "Pretty good for the price.",
            "Completely broke after one use.",
            "Exceeded my expectations!",
            "Average quality, nothing impressive.",
            "Absolutely love it, will buy again!",
            "Disappointed, not as described.",
        ]})
        st.download_button("📥 Sample CSV", data=sample.to_csv(index=False),
                           file_name="sample_reviews.csv", mime="text/csv",
                           use_container_width=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("❌ CSV must have a 'text' column.")
        else:
            st.success(f"✅ Loaded **{len(df)}** rows!")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Analysis", type="primary"):
                texts = df["text"].astype(str).tolist()[:100]
                prog  = st.progress(0, text="🧠 Analyzing...")
                with st.spinner("Processing..."):
                    raw = classifier(texts)
                prog.progress(100, text="✅ Done!")
                time.sleep(0.3)
                prog.empty()

                results_df = pd.DataFrame([
                    {"text": t, "label": r["label"], "confidence": round(r["score"]*100, 1)}
                    for t, r in zip(texts, raw)
                ])
                pos   = sum(1 for r in raw if r["label"] == "POSITIVE")
                neg   = len(raw) - pos
                avg_c = results_df["confidence"].mean()

                st.divider()
                st.markdown("### 📊 Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📝 Total",          len(raw))
                m2.metric("😊 Positive",       pos)
                m3.metric("😞 Negative",       neg)
                m4.metric("🎯 Avg Confidence", f"{avg_c:.1f}%")

                ch1, ch2, ch3 = st.columns(3)

                with ch1:
                    pie = go.Figure(go.Pie(
                        labels=["POSITIVE", "NEGATIVE"], values=[pos, neg],
                        marker=dict(colors=[GREEN, RED],
                                    line=dict(color="#0f0c29", width=2)),
                        hole=0.5, textfont=dict(color="white"),
                    ))
                    pie.update_layout(**plot_layout("Sentiment Split"))
                    st.plotly_chart(pie, use_container_width=True)

                with ch2:
                    hist = px.histogram(
                        results_df, x="confidence", color="label",
                        color_discrete_map={"POSITIVE": GREEN, "NEGATIVE": RED},
                        nbins=20, barmode="overlay", opacity=0.8,
                    )
                    hist.update_layout(**plot_layout("Confidence Distribution"))
                    hist.update_xaxes(showgrid=False)
                    hist.update_yaxes(showgrid=False)
                    st.plotly_chart(hist, use_container_width=True)

                with ch3:
                    box = px.box(
                        results_df, x="label", y="confidence", color="label",
                        color_discrete_map={"POSITIVE": GREEN, "NEGATIVE": RED},
                    )
                    box.update_layout(**plot_layout("Confidence by Label"))
                    box.update_xaxes(showgrid=False)
                    box.update_yaxes(showgrid=False)
                    st.plotly_chart(box, use_container_width=True)

                st.markdown("### 📋 Results Table")
                st.dataframe(
                    results_df.style.applymap(
                        lambda v: f"color: {GREEN}" if v=="POSITIVE" else f"color: {RED}",
                        subset=["label"],
                    ).background_gradient(subset=["confidence"], cmap="Purples"),
                    use_container_width=True, height=320,
                )
                st.download_button(
                    "📥 Download Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )
