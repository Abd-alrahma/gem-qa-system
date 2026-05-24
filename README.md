# 🏛️ Grand Egyptian Museum — Arabic QA System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=for-the-badge&logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple?style=for-the-badge)
![Selenium](https://img.shields.io/badge/Selenium-Chrome-yellow?style=for-the-badge&logo=selenium)

**A full end-to-end Arabic Website Question Answering system built on the official Grand Egyptian Museum website ([gem.eg/ar](https://gem.eg/ar))**

[Features](#-features) • [Pipeline](#-pipeline) • [Models](#-models-used) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-reference) • [Project Structure](#-project-structure)

</div>

---

## 📌 Project Description

This project is a **Graduation Project** that implements a complete NLP pipeline for Arabic Question Answering over real website content.

The system:
1. **Scrapes** the official GEM Arabic website using Selenium (JavaScript-rendered)
2. **Chunks** content by section (heading + paragraphs) for coherent retrieval
3. **Embeds** chunks using a multilingual model that natively supports Arabic
4. **Indexes** vectors in FAISS for millisecond similarity search
5. **Answers** user questions in Arabic via a REST API and web interface

> **Why this is hard:** gem.eg is a JavaScript-rendered (React/Next.js) website that blocks simple HTTP requests with 403. All text is in Arabic. Standard English-only models fail completely. This project solves all three challenges.

---

## ✨ Features

- 🌐 **Real website data** — scraped from the official GEM Arabic website
- 🔍 **Semantic search** — finds answers by meaning, not keyword matching
- 🇦🇷 **Full Arabic support** — RTL frontend, Arabic embeddings, Arabic QA model
- ⚡ **Three answer modes** — Extractive, Generative, and Auto
- 🖥️ **Web interface** — clean Arabic RTL frontend with sample questions
- 📡 **REST API** — FastAPI with automatic documentation at `/docs`
- 🆓 **100% free** — no paid API keys required, all models run locally

---

## 🔄 Pipeline

```
gem.eg/ar
    │
    ▼
┌─────────────────┐
│   Selenium      │  Opens Chrome invisibly, renders JavaScript
│   Scraper       │  Extracts Arabic text section by section
└────────┬────────┘
         │ chunks.json
         ▼
┌─────────────────┐
│  Section-Based  │  heading + paragraphs = one chunk
│   Chunking      │  Max 120 words, 10-word overlap, noise filtered
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ multilingual-   │  "passage: " + chunk → 768-dim vector
│   e5-base       │  Supports Arabic + 100 languages
└────────┬────────┘
         │ embeddings
         ▼
┌─────────────────┐
│  FAISS Index    │  IndexFlatIP — exact cosine similarity search
│  (IndexFlatIP)  │  Stores all chunk vectors
└────────┬────────┘
         │
    User Question
         │
         ▼
┌─────────────────┐
│  Query Embed    │  "query: " + question → 768-dim vector
│  + FAISS Search │  Returns top-K most similar chunks
└────────┬────────┘
         │
    ┌────┴──────────────┐
    │                   │
    ▼                   ▼
Extractive          Generative
(1 best chunk)    (top-3 merged)
    │                   │
    └────────┬──────────┘
             │
             ▼
┌─────────────────┐
│   FastAPI       │  GET /ask?question=...&mode=...
│   REST API      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Arabic HTML    │  RTL interface, 3 mode buttons
│  Frontend       │  Confidence score display
└─────────────────┘
```

---

## 🤖 Models Used

| Model | Role | Why This Model |
|-------|------|----------------|
| `intfloat/multilingual-e5-base` | Text Embedding | Supports Arabic natively. Requires `query:` / `passage:` prefix. Better than English-only models. |
| `ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA` | Arabic QA (Extractive) | Fine-tuned on Arabic SQuAD. Understands Arabic question-answer structure. |
| `FAISS IndexFlatIP` | Vector Search | Exact cosine similarity. Fast for datasets under 100K vectors. No approximation errors. |
| `Selenium + ChromeDriver` | Web Scraping | gem.eg is JavaScript-rendered. Requests alone return 403. Selenium renders pages like a real browser. |

### Why NOT these models?

| Rejected Model | Reason |
|----------------|--------|
| `all-MiniLM-L6-v2` | English-only — produces poor vectors for Arabic |
| `requests + BeautifulSoup` | Cannot execute JavaScript — returns empty HTML |
| `FAISS IndexIVFFlat` | Approximate search — unnecessary for small datasets |
| `gpt-3.5 / gpt-4` | Paid API — project requirement was free local models |

---

## 📁 Project Structure

```
gem-qa-system/
│
├── scraper.py          # Selenium scraper — crawls gem.eg/ar
├── build_index.py      # Builds FAISS index from chunks.json
├── api.py              # FastAPI backend — serves QA answers
├── index.html          # Arabic RTL frontend
├── requirements.txt    # Python dependencies
│
├── chunks.json         # Scraped Arabic text chunks (generated)
├── faiss_index.index   # FAISS vector index (generated)
│
└── README.md
```

> `chunks.json` and `faiss_index.index` are generated by running `scraper.py` and `build_index.py`. They are not included in the repository.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Google Chrome installed
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/gem-qa-system.git
cd gem-qa-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Scrape the website (opens Chrome invisibly, ~5-10 minutes)
python scraper.py

# 4. Build the FAISS index (downloads ~1GB model on first run)
python build_index.py

# 5. Start the API
uvicorn api:app --reload

# 6. Open the frontend
# Just double-click index.html in File Explorer
# Or open: http://127.0.0.1:8000/docs for API documentation
```

---

## 🚀 Usage

### Web Interface

Open `index.html` in your browser while the API is running.

**Three answer modes:**

| Mode | Arabic | Best For |
|------|--------|----------|
| Extractive | استخراجي | Short precise answers — names, dates, numbers |
| Generative | توليدي | Detailed answers — why, how, explain |
| Auto | تلقائي | Smart selection based on confidence score |

**Sample questions to try:**
- `ما هو المتحف المصري الكبير؟`
- `أين يقع المتحف المصري الكبير؟`
- `ما هي ساعات العمل؟`
- `ما هي مجموعة توت عنخ آمون؟`
- `كيف أصل إلى المتحف؟`

---

## 📡 API Reference

### `GET /ask`

Ask a question about the GEM website.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | Your question in Arabic |
| `mode` | string | `auto` | `extractive` \| `generative` \| `auto` |
| `top_k` | integer | `10` | Number of chunks to retrieve (1–30) |

**Example Request:**
```
GET http://127.0.0.1:8000/ask?question=أين يقع المتحف؟&mode=extractive
```

**Example Response (Extractive):**
```json
{
  "mode": "extractive",
  "answer": "يقع المتحف المصري الكبير على طريق القاهرة - الإسكندرية الصحراوي، ميدان الرماية، الجيزة، مصر",
  "confidence": 0.8612
}
```

**Example Response (Generative):**
```json
{
  "mode": "generative",
  "answer": "يقع المتحف المصري الكبير على طريق القاهرة - الإسكندرية الصحراوي...",
  "retrieval_score": 0.8612,
  "sources_used": 3
}
```

### `GET /health`

Check that the API is running and all models are loaded.

```json
{
  "status": "ok",
  "chunks": 245,
  "vectors": 245,
  "model": "intfloat/multilingual-e5-base",
  "approach": "retrieval-based"
}
```

### Interactive Docs

FastAPI generates automatic interactive documentation:
```
http://127.0.0.1:8000/docs
```

---

## 📊 Answer Mode Logic

```
User Question
     │
     ▼
FAISS Retrieval → top_k chunks with scores
     │
     ├── mode = "extractive" → return best single chunk
     │
     ├── mode = "generative" → merge top-3 chunks (score ≥ 0.55)
     │
     └── mode = "auto"
              │
              ├── score ≥ 0.70 → extractive (high confidence)
              └── score < 0.70 → generative (broader coverage)
```

**Score Interpretation:**

| Score | Meaning |
|-------|---------|
| > 0.70 | Excellent match — highly relevant answer |
| 0.55 – 0.70 | Good match — related answer |
| < 0.55 | Weak match — question may be outside website content |

---

## 🛠️ Technical Decisions

### Why Section-Based Chunking?
Word-count chunking (every N words) mixes unrelated content in one chunk — location + opening hours + privacy policy all in one block. Section-based chunking groups each heading with its paragraphs, so one chunk = one topic = one accurate answer.

### Why Retrieval-Based Answering?
Span extraction models (AraElectra, RoBERTa) are trained on clean SQuAD-style paragraphs. Real website content is much noisier — mixed topics, repeated navigation text, incomplete sentences. Returning the full retrieved chunk as the answer is more reliable and always on-topic.

### Why multilingual-e5 with Prefixes?
The multilingual-e5 architecture requires specific prefixes:
- Chunks: `"passage: " + chunk_text`
- Questions: `"query: " + question_text`

Without these prefixes, retrieval quality drops significantly.

---

## 🚧 Limitations

- Answers are bounded by website content — cannot answer questions not on gem.eg/ar
- JavaScript-heavy pages require longer scraping time (~5–10 minutes)
- Answers are returned as full paragraphs — not always a single concise sentence
- Model downloads require ~1GB on first run

## 🔮 Future Improvements

- [ ] Integrate a generative LLM (Llama 3) for concise synthesized answers
- [ ] Add cross-encoder re-ranker for higher precision
- [ ] Expand to multiple Egyptian websites (Al-Azhar, Dar Al-Ifta, EgyptAir)
- [ ] Add answer caching for frequent questions
- [ ] Deploy to cloud (Render, Railway, or Hugging Face Spaces)

---

## 📋 Requirements

```
fastapi
uvicorn
faiss-cpu
sentence-transformers
transformers
torch
selenium
webdriver-manager
beautifulsoup4
requests
```

---

## 👤 Author

Built as a Graduation Project — Natural Language Processing (NLP)

---

## 📄 License

This project is for educational purposes only.
Website content belongs to the Grand Egyptian Museum (gem.eg).
