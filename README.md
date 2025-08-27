# arxiv-top-papers

A **Streamlit web app** to find and rank top arXiv papers in **Reinforcement Learning (RL)** and **Natural Language Processing (NLP)**. The tool queries the [arXiv API](https://arxiv.org/help/api/), optionally enriches results with [OpenAlex](https://openalex.org) citation counts, and ranks them based on recency, citations, and keyword relevance.

---

## 🚀 Features
- Query arXiv by keywords and categories (RL/NLP presets included).
- Adjustable **lookback window** (last N months).
- Ranking by:
  - **Recency** (exponential decay with half-life)
  - **Citations** (OpenAlex enrichment)
  - **Keyword relevance**
- Interactive sidebar for weights and filters.
- Download results as **CSV**.
- Clean paper cards with titles, authors, abstracts, links, and scores.

---

## 📂 Project Structure
```
arxiv-top-papers/
├── app/
│   └── arxiv_top_papers_app.py   # Main Streamlit app
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

---

## 🛠 Installation

Clone the repo:
```bash
git clone https://github.com/yourusername/arxiv-top-papers.git
cd arxiv-top-papers
```

Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the Streamlit app:
```bash
streamlit run app/arxiv_top_papers_app.py
```

Open the provided local URL (default: `http://localhost:8501`) in your browser.

---

## 📦 requirements.txt
```
streamlit
feedparser
requests
```

---

## 📝 Example Search Flow
1. Choose **Reinforcement Learning** preset in the sidebar.
2. Set lookback to **6 months**.
3. Enable **OpenAlex enrichment** for citation counts.
4. Adjust weights: Recency=1.0, Citations=1.0, Relevance=0.5.
5. Click **Search & Rank**.
6. Browse top K papers, download CSV.

---

## 🔮 Roadmap
- [ ] Add presets for RLHF, LLM Agents, Multimodal.
- [ ] Advanced filters (exclude surveys, include benchmarks).
- [ ] Add author/venue filtering.
- [ ] Deploy to Hugging Face Spaces.

---

## 📜 License
MIT License.

---

## 🙌 Acknowledgements
- [arXiv API](https://arxiv.org/help/api/)
- [OpenAlex](https://openalex.org)
- [Streamlit](https://streamlit.io)


### `.gitignore`
```
.venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/
```
