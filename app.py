"""
SentimentIQ — Neon Green & Dark Theme
Features: Word Cloud, History Log, Emoji Animation, Bold Results, New Charts
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from collections import Counter
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
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #080c0e;
    min-height: 100vh;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    background: linear-gradient(135deg, rgba(0,255,136,0.06), rgba(0,200,100,0.03));
    border-radius: 20px;
    border: 1px solid rgba(0,255,136,0.25);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,255,136,0.04) 0%, transparent 60%);
    animation: rotateBg 8s linear infinite;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #00ff88;
    text-shadow: 0 0 30px rgba(0,255,136,0.6), 0 0 60px rgba(0,255,136,0.3);
    margin-bottom: 0.4rem;
    letter-spacing: -1px;
    position: relative;
}
.hero p { color: #4a7c59; font-size: 0.95rem; position: relative; }

/* ── Result MEGA display ── */
.mega-result {
    text-align: center;
    padding: 2.5rem;
    border-radius: 20px;
    margin: 1.5rem 0;
    animation: popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}
.mega-result.positive {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,200,100,0.06));
    border: 2px solid rgba(0,255,136,0.5);
    box-shadow: 0 0 40px rgba(0,255,136,0.15), inset 0 0 40px rgba(0,255,136,0.03);
}
.mega-result.negative {
    background: linear-gradient(135deg, rgba(255,50,50,0.12), rgba(200,0,50,0.06));
    border: 2px solid rgba(255,80,80,0.5);
    box-shadow: 0 0 40px rgba(255,50,50,0.15), inset 0 0 40px rgba(255,50,50,0.03);
}
.mega-emoji {
    font-size: 5rem;
    animation: bounceEmoji 0.8s ease;
    display: block;
    margin-bottom: 0.5rem;
}
.mega-label {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
}
.mega-result.positive .mega-label { color: #00ff88; text-shadow: 0 0 20px rgba(0,255,136,0.7); }
.mega-result.negative .mega-label { color: #ff5050; text-shadow: 0 0 20px rgba(255,80,80,0.7); }
.mega-score {
    font-size: 1.2rem;
    color: #4a7c59;
    margin-top: 0.5rem;
    font-family: 'Share Tech Mono', monospace;
}

/* ── Stat cards ── */
.stat-card {
    background: rgba(0,255,136,0.04);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: rgba(0,255,136,0.5);
    box-shadow: 0 0 20px rgba(0,255,136,0.1);
    transform: translateY(-2px);
}
.stat-label { font-size: 0.72rem; color: #4a7c59; text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-size: 1.6rem; font-weight: 700; color: #00ff88; margin-top: 0.3rem; }

/* ── History log ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.75rem 1rem;
    background: rgba(0,255,136,0.03);
    border: 1px solid rgba(0,255,136,0.12);
    border-radius: 10px;
    margin-bottom: 0.5rem;
    animation: slideRight 0.3s ease;
    font-size: 0.85rem;
}
.history-badge-pos {
    background: rgba(0,255,136,0.2);
    color: #00ff88;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
}
.history-badge-neg {
    background: rgba(255,80,80,0.2);
    color: #ff5050;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
}
.history-text { color: #94a3b8; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.history-conf { color: #4a7c59; font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; white-space: nowrap; }

/* ── Buttons ── */
.stButton > button {
    background: rgba(0,255,136,0.06) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
    color: #00ff88 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,0.15) !important;
    box-shadow: 0 0 15px rgba(0,255,136,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c864, #00ff88) !important;
    border: none !important;
    color: #080c0e !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(0,255,136,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 35px rgba(0,255,136,0.55) !important;
    transform: translateY(-2px) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid rgba(0,255,136,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 0 2px rgba(0,255,136,0.15) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060a0c !important;
    border-right: 1px solid rgba(0,255,136,0.15) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,255,136,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(0,255,136,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    color: #4a7c59 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,255,136,0.15) !important;
    color: #00ff88 !important;
    box-shadow: 0 0 10px rgba(0,255,136,0.2) !important;
}

hr { border-color: rgba(0,255,136,0.15) !important; }

/* ── Animations ── */
@keyframes popIn {
    0%   { opacity:0; transform: scale(0.8); }
    70%  { transform: scale(1.03); }
    100% { opacity:1; transform: scale(1); }
}
@keyframes bounceEmoji {
    0%   { transform: scale(0) rotate(-20deg); }
    60%  { transform: scale(1.2) rotate(5deg); }
    100% { transform: scale(1) rotate(0); }
}
@keyframes slideRight {
    from { opacity:0; transform: translateX(-15px); }
    to   { opacity:1; transform: translateX(0); }
}
@keyframes rotateBg {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes neonPulse {
    0%,100% { text-shadow: 0 0 20px rgba(0,255,136,0.6); }
    50%     { text-shadow: 0 0 40px rgba(0,255,136,1), 0 0 80px rgba(0,255,136,0.4); }
}
</style>
""", unsafe_allow_html=True)

# ── Plot helpers ──────────────────────────────────────────────────────────────
NEON    = "#00ff88"
RED     = "#ff5050"
DARK_BG = "rgba(6,10,12,0.9)"
FONT_C  = "#94a3b8"

def plot_layout(title="", h=300):
    return dict(
        title=dict(text=title, font=dict(color=NEON, size=13, family="Inter")),
        paper_bgcolor=DARK_BG,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_C, family="Inter"),
        height=h,
        margin=dict(l=20, r=20, t=45, b=20),
    )

# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Thinzar2003/sentiment-distilbert")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0 1rem;'>
        <div style='font-size:2.8rem; filter:drop-shadow(0 0 10px #00ff88)'>🧠</div>
        <div style='font-size:1.3rem; font-weight:800; color:#00ff88 !important;
                    text-shadow:0 0 15px rgba(0,255,136,0.5);'>SentimentIQ</div>
        <div style='font-size:0.72rem; color:#4a7c59 !important;'>NLP Portfolio Project</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='color:#00ff88 !important; font-weight:600; font-size:0.85rem'>⚡ HISTORY</div>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("<div style='color:#4a7c59 !important; font-size:0.8rem; padding:0.5rem 0'>No analyses yet...</div>", unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history[-8:]):
            badge_cls = "history-badge-pos" if item["label"] == "POSITIVE" else "history-badge-neg"
            emoji     = "😊" if item["label"] == "POSITIVE" else "😞"
            st.markdown(f"""
            <div class='history-item'>
                <span class='{badge_cls}'>{emoji} {item['label']}</span>
                <span class='history-text'>{item['text'][:35]}...</span>
                <span class='history-conf'>{item['conf']}%</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.82rem; line-height:2; color:#4a7c59 !important;'>
    📦 <b style='color:#00ff88 !important'>Model</b><br>
    DistilBERT (fine-tuned)<br><br>
    📚 <b style='color:#00ff88 !important'>Dataset</b><br>
    IMDB 50k reviews<br><br>
    🎯 <b style='color:#00ff88 !important'>Accuracy</b><br>
    ~93% F1 Score
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>⚡ SentimentIQ</h1>
    <p>Real-time AI sentiment analysis · Fine-tuned DistilBERT · HuggingFace + Streamlit</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("⚡ Booting AI engine..."):
    classifier = load_model()

tab1, tab2 = st.tabs(["🔤  Analyze Text", "📂  Batch CSV"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Text
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    examples = [
        "This movie was absolutely fantastic! A total masterpiece.",
        "Terrible film. Complete waste of 2 hours, extremely boring.",
        "It was alright, had some good moments but nothing special.",
    ]

    st.markdown("##### ⚡ Quick Examples")
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex[:40] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["input_text"] = ex

    user_text = st.text_area(
        "✍️ Enter text to analyze:",
        value=st.session_state.get("input_text", ""),
        placeholder="Type a review, tweet, feedback, or any text...",
        height=120,
        key="input_text",
    )

    st.markdown("")
    if st.button("⚡ ANALYZE SENTIMENT", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🧠 Thinking..."):
                t0     = time.time()
                result = classifier(user_text)[0]
                latency = round((time.time() - t0) * 1000, 1)

            label     = result["label"]
            score     = result["score"]
            conf      = round(score * 100, 1)
            emoji     = "😊" if label == "POSITIVE" else "😱"
            cls       = "positive" if label == "POSITIVE" else "negative"
            bar_color = NEON if label == "POSITIVE" else RED

            # Save to history
            st.session_state.history.append({
                "text": user_text, "label": label,
                "conf": conf, "latency": latency,
            })

            # ── MEGA result display ───────────────────────────────────────
            st.markdown(f"""
            <div class='mega-result {cls}'>
                <span class='mega-emoji'>{emoji}</span>
                <div class='mega-label'>{label}</div>
                <div class='mega-score'>confidence: {conf}% &nbsp;|&nbsp; latency: {latency}ms</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Stat cards ────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Sentiment</div><div class='stat-value'>{emoji}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Confidence</div><div class='stat-value'>{conf}%</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Latency</div><div class='stat-value'>{latency}ms</div></div>", unsafe_allow_html=True)
            with c4:
                total = len(st.session_state.history)
                st.markdown(f"<div class='stat-card'><div class='stat-label'>Total Analyzed</div><div class='stat-value'>{total}</div></div>", unsafe_allow_html=True)

            st.markdown("")

            # ── Charts ────────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)

            with ch1:
                # Radial gauge
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=conf,
                    delta={"reference": 50, "increasing": {"color": NEON}, "decreasing": {"color": RED}},
                    number={"suffix": "%", "font": {"color": bar_color, "size": 32, "family": "Share Tech Mono"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": NEON, "tickfont": {"color": FONT_C}},
                        "bar": {"color": bar_color, "thickness": 0.3},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "rgba(0,255,136,0.2)",
                        "steps": [
                            {"range": [0,  40], "color": "rgba(255,80,80,0.1)"},
                            {"range": [40, 70], "color": "rgba(255,200,0,0.07)"},
                            {"range": [70,100], "color": "rgba(0,255,136,0.1)"},
                        ],
                        "threshold": {
                            "line": {"color": NEON, "width": 2},
                            "thickness": 0.8, "value": 80,
                        },
                    },
                    title={"text": "Confidence Gauge", "font": {"color": NEON, "size": 13}},
                ))
                gauge.update_layout(**plot_layout(h=270))
                st.plotly_chart(gauge, use_container_width=True)

            with ch2:
                # Horizontal bar — pos vs neg
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="POSITIVE", x=[score if label=="POSITIVE" else 1-score],
                    y=["Score"], orientation="h",
                    marker=dict(color=NEON, opacity=0.85),
                    text=[f"😊 {(score if label=='POSITIVE' else 1-score)*100:.1f}%"],
                    textposition="inside", insidetextfont=dict(color="#080c0e", size=13),
                ))
                fig_bar.add_trace(go.Bar(
                    name="NEGATIVE", x=[score if label=="NEGATIVE" else 1-score],
                    y=["Score"], orientation="h",
                    marker=dict(color=RED, opacity=0.85),
                    text=[f"😞 {(score if label=='NEGATIVE' else 1-score)*100:.1f}%"],
                    textposition="inside", insidetextfont=dict(color="white", size=13),
                ))
                fig_bar.update_layout(
                    **plot_layout("Positive vs Negative Score", h=270),
                    barmode="stack",
                    xaxis=dict(range=[0,1], showgrid=False, visible=False),
                    yaxis=dict(showgrid=False),
                    legend=dict(orientation="h", y=-0.15, font=dict(color=FONT_C)),
                    showlegend=True,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Word frequency chart ──────────────────────────────────────
            st.markdown("#### 🔤 Word Frequency")
            stopwords = {"the","a","an","is","it","in","on","at","to","of","and","or",
                         "but","for","with","this","that","was","i","my","me","we","are","be"}
            words = [w.lower().strip(".,!?\"'") for w in user_text.split()
                     if len(w) > 2 and w.lower().strip(".,!?\"'") not in stopwords]

            if words:
                word_counts = Counter(words).most_common(12)
                wdf = pd.DataFrame(word_counts, columns=["word", "count"])
                wfig = px.bar(
                    wdf, x="count", y="word", orientation="h",
                    color="count",
                    color_continuous_scale=[[0, "#003322"], [0.5, "#00cc66"], [1, "#00ff88"]],
                )
                wfig.update_layout(
                    **plot_layout("Word Frequency in Your Text", h=max(200, len(words)*22)),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False, autorange="reversed"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(wfig, use_container_width=True)

            # ── Analyzed text ─────────────────────────────────────────────
            st.markdown(f"""
            <div style='background:rgba(0,255,136,0.04); border:1px solid rgba(0,255,136,0.2);
                        border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;'>
                <div style='color:#4a7c59; font-size:0.72rem; margin-bottom:0.4rem; letter-spacing:0.1em'>ANALYZED TEXT</div>
                <div style='color:#e2e8f0; font-size:0.95rem; line-height:1.7'>"{user_text}"</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Session history chart ─────────────────────────────────────────────────
    if len(st.session_state.history) >= 2:
        st.divider()
        st.markdown("#### 📈 Session History")
        hdf = pd.DataFrame(st.session_state.history)
        hdf["index"] = range(1, len(hdf) + 1)
        hdf["color"] = hdf["label"].map({"POSITIVE": NEON, "NEGATIVE": RED})

        hfig = go.Figure()
        hfig.add_trace(go.Scatter(
            x=hdf["index"], y=hdf["conf"],
            mode="lines+markers",
            line=dict(color=NEON, width=2, dash="dot"),
            marker=dict(
                color=hdf["color"].tolist(), size=12,
                line=dict(color="#080c0e", width=2),
            ),
            text=hdf["label"],
            hovertemplate="<b>%{text}</b><br>Confidence: %{y}%<extra></extra>",
        ))
        hfig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.1)")
        hfig.update_layout(
            **plot_layout("Confidence Over Session", h=250),
            xaxis=dict(showgrid=False, title="Analysis #", tickcolor=FONT_C),
            yaxis=dict(showgrid=False, range=[0, 105], title="Confidence %"),
        )
        st.plotly_chart(hfig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch CSV
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

            if st.button("⚡ RUN BATCH ANALYSIS", type="primary"):
                texts = df["text"].astype(str).tolist()[:100]
                prog  = st.progress(0, text="🧠 Analyzing batch...")
                with st.spinner("Processing..."):
                    raw = classifier(texts)
                prog.progress(100, text="✅ Complete!")
                time.sleep(0.3)
                prog.empty()

                results_df = pd.DataFrame([
                    {"text": t, "label": r["label"], "confidence": round(r["score"]*100, 1)}
                    for t, r in zip(texts, raw)
                ])
                pos   = sum(1 for r in raw if r["label"] == "POSITIVE")
                neg   = len(raw) - pos
                avg_c = results_df["confidence"].mean()

                # Mega summary cards
                st.markdown(f"""
                <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:1.5rem 0;'>
                    <div class='stat-card'><div class='stat-label'>Total</div><div class='stat-value'>{len(raw)}</div></div>
                    <div class='stat-card'><div class='stat-label'>Positive 😊</div><div class='stat-value' style='color:#00ff88'>{pos}</div></div>
                    <div class='stat-card'><div class='stat-label'>Negative 😞</div><div class='stat-value' style='color:#ff5050'>{neg}</div></div>
                    <div class='stat-card'><div class='stat-label'>Avg Confidence</div><div class='stat-value'>{avg_c:.1f}%</div></div>
                </div>
                """, unsafe_allow_html=True)

                ch1, ch2 = st.columns(2)

                with ch1:
                    # Donut
                    donut = go.Figure(go.Pie(
                        labels=["POSITIVE 😊", "NEGATIVE 😞"],
                        values=[pos, neg],
                        marker=dict(colors=[NEON, RED],
                                    line=dict(color="#080c0e", width=3)),
                        hole=0.55,
                        textfont=dict(color="white", size=13),
                        hovertemplate="%{label}: %{value}<br>%{percent}<extra></extra>",
                    ))
                    donut.add_annotation(
                        text=f"{round(pos/len(raw)*100)}%<br><span style='font-size:10px'>positive</span>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=18, color=NEON, family="Inter"),
                    )
                    donut.update_layout(**plot_layout("Sentiment Split"))
                    st.plotly_chart(donut, use_container_width=True)

                with ch2:
                    # Confidence histogram
                    hist = px.histogram(
                        results_df, x="confidence", color="label",
                        color_discrete_map={"POSITIVE": NEON, "NEGATIVE": RED},
                        nbins=20, barmode="overlay", opacity=0.75,
                    )
                    hist.update_layout(**plot_layout("Confidence Distribution"))
                    hist.update_xaxes(showgrid=False, title="Confidence %")
                    hist.update_yaxes(showgrid=False, title="Count")
                    st.plotly_chart(hist, use_container_width=True)

                # Scatter confidence by index
                scatter = px.scatter(
                    results_df.reset_index(), x="index", y="confidence",
                    color="label",
                    color_discrete_map={"POSITIVE": NEON, "NEGATIVE": RED},
                    size="confidence", size_max=14,
                    hover_data={"text": True, "label": True, "confidence": True, "index": False},
                )
                scatter.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.1)")
                scatter.update_layout(
                    **plot_layout("Per-Review Confidence Scatter", h=300),
                    xaxis=dict(showgrid=False, title="Review #"),
                    yaxis=dict(showgrid=False, title="Confidence %", range=[0,105]),
                )
                st.plotly_chart(scatter, use_container_width=True)

                # Word frequency across all texts
                st.markdown("#### 🔤 Top Words Across All Reviews")
                stopwords = {"the","a","an","is","it","in","on","at","to","of","and","or",
                             "but","for","with","this","that","was","i","my","me","we","are","be","not"}
                all_words = []
                for t in texts:
                    all_words += [w.lower().strip(".,!?\"'") for w in t.split()
                                  if len(w) > 2 and w.lower() not in stopwords]
                wc = Counter(all_words).most_common(15)
                wdf = pd.DataFrame(wc, columns=["word", "count"])
                wfig = px.bar(
                    wdf, x="word", y="count",
                    color="count",
                    color_continuous_scale=[[0,"#003322"],[0.5,"#00cc66"],[1,"#00ff88"]],
                )
                wfig.update_layout(
                    **plot_layout("Most Frequent Words", h=280),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(wfig, use_container_width=True)

                # Results table
                st.markdown("### 📋 Full Results")
                st.dataframe(
                    results_df.style.applymap(
                        lambda v: f"color: {NEON}" if v=="POSITIVE" else f"color: {RED}",
                        subset=["label"],
                    ).background_gradient(subset=["confidence"], cmap="Greens"),
                    use_container_width=True, height=320,
                )
                st.download_button(
                    "📥 Download Results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )
