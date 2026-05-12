# 💬 Sentiment Analysis Dashboard

An end-to-end NLP portfolio project: fine-tune DistilBERT on IMDB reviews, serve predictions via FastAPI, and visualize results in a Streamlit web app.

**Live Demo**: [your-app.streamlit.app](https://sentiment-analysis-dashboard-dqsqvxqotjpkswwvhcyldq.streamlit.app/) 

---

## 🏗️ Architecture

```
Data (IMDB) → Fine-tune DistilBERT → Save model
                                          ↓
                                    FastAPI /predict
                                          ↓
                                  Streamlit Dashboard
```

## 📁 Project Structure

```
sentiment-dashboard/
├── data/                  # EDA plots, sample CSVs
├── notebooks/
│   └── train_sentiment_model.ipynb   # Full training pipeline (Google Colab)
├── model/
│   └── sentiment-distilbert/         # Saved model weights (after training)
├── api/
│   └── main.py            # FastAPI backend
├── app/
│   └── app.py             # Streamlit frontend
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Train the model (Google Colab)
Open `notebooks/train_sentiment_model.ipynb` in Google Colab.  
Enable GPU: **Runtime → Change runtime type → T4 GPU**  
Run all cells. Model saves to `model/sentiment-distilbert/`.

### 2. Run the API locally
```bash
pip install -r requirements.txt
cd api
uvicorn main:app --reload --port 8000
```
Visit: http://localhost:8000/docs

### 3. Run the Streamlit app
```bash
cd app
streamlit run app.py
```
Visit: http://localhost:8501

---

## 🌐 Deploy for Free

### Option A: Hugging Face Spaces (Recommended)
1. Push model weights to HF Hub: `model.push_to_hub("your-username/sentiment-distilbert")`
2. Create a new Space at huggingface.co/spaces
3. Choose **Streamlit** SDK, upload `app/app.py` and `requirements.txt`

### Option B: Streamlit Community Cloud
1. Push repo to GitHub (public)
2. Go to share.streamlit.io → New App
3. Select repo, set main file to `app/app.py`

---

## 📊 Model Performance

| Metric   | Score  |
|----------|--------|
| Accuracy | ~93%   |
| F1 Score | ~0.93  |
| Model    | DistilBERT-base-uncased |
| Dataset  | IMDB (4k train / 1k test) |

> Trained on 4k samples for speed. Retrain on full 50k dataset for higher accuracy.

---

## 🛠️ Tech Stack

| Layer     | Technology |
|-----------|-----------|
| Model     | HuggingFace Transformers (DistilBERT) |
| Training  | PyTorch + HuggingFace Trainer API |
| Backend   | FastAPI + Uvicorn |
| Frontend  | Streamlit + Plotly |
| Deployment| HuggingFace Spaces / Streamlit Cloud |

---

## 💡 Features

- ✅ Single text sentiment prediction with confidence score
- ✅ Batch CSV upload (up to 100 rows)
- ✅ Interactive charts: pie chart + confidence histogram
- ✅ Downloadable results CSV
- ✅ REST API with `/predict` and `/predict/batch` endpoints
- ✅ API health check endpoint

---

## 🔮 Future Improvements

- [ ] Add neutral sentiment (3-class model)
- [ ] Support multilingual reviews
- [ ] Add SHAP explainability (which words drive the prediction)
- [ ] Persist history with SQLite
- [ ] Docker deployment

---

*Built as a portfolio project showcasing end-to-end NLP with HuggingFace, FastAPI, and Streamlit.*
