# QuickLisis

> AI-powered data analysis platform that turns CSV and Excel datasets into actionable insights.

## Live Demo

**Frontend:** https://quicklisis.vercel.app

**Backend API:** https://quicklisis.onrender.com

**Health Check:** https://quicklisis.onrender.com/api/health

---

## Overview

QuickLisis helps users upload a dataset, understand its structure and quality, perform analysis, visualize data, and ask questions about the dataset using natural language.

The application combines **Pandas for deterministic data analysis** with **Google Gemini for natural-language understanding and explanation**.

### Core workflow

```text
Upload CSV / XLSX
        ↓
Dataset Profiling
        ↓
Data Quality
        ↓
Data Cleaning
        ↓
EDA & Statistics
        ↓
KPIs & Charts
        ↓
AI Data Analyst
        ↓
Natural-Language Insights
```

---

## Features

- CSV and XLSX file upload
- File validation and size limits
- Automatic Excel header detection
- Dataset profiling
- Column and data-type analysis
- Missing-value analysis
- Unique-value analysis
- Statistical summaries
- Data cleaning workflow
- Exploratory Data Analysis
- KPI and analytical calculations
- Chart-data generation
- Paginated dataset preview
- Natural-language data questions
- Gemini-powered AI analyst
- AI result review and validation
- REST API backend
- Production deployment

---

## AI Data Analyst

Users can ask questions about their uploaded dataset using natural language.

Examples:

```text
What is the average salary?

Which department has the highest average salary?

What are the top products by revenue?

Which region generated the highest sales?

What is the oldest employee?

Show me the monthly trend.
```

QuickLisis uses a controlled AI analysis pipeline:

```text
User Question
      ↓
Gemini Planner
      ↓
Controlled Analysis Plan
      ↓
Pandas Executor
      ↓
Calculated Result
      ↓
Gemini Reviewer
      ↓
Natural-Language Answer
```

The LLM does not directly execute generated Python code. Analytical operations are validated and executed by the backend using Pandas.

This separates:

- **AI:** understanding, planning and explanation
- **Pandas:** data processing and calculation

---

## Tech Stack

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn
- Pandas
- NumPy
- Requests

### AI

- Google Gemini API
- Structured JSON analytical planning
- AI result review

### Deployment & Tools

- Git
- GitHub
- Vercel
- Render
- Environment Variables

### Database Foundation

- SQLite

---

## Architecture

```text
                    ┌──────────────────────┐
                    │      React UI        │
                    │ TypeScript + Tailwind│
                    └──────────┬───────────┘
                               │
                               │ REST API
                               ▼
                    ┌──────────────────────┐
                    │       FastAPI        │
                    │       Backend        │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
           Pandas            NumPy         Gemini API
              │                                 │
              └──────── Data Analysis ──────────┘
```

### Production

```text
GitHub
  ├── Frontend → Vercel
  └── Backend  → Render
                    │
                    └── Google Gemini API
```

---

## API

Main API endpoints include:

```text
GET  /api/health

POST /api/datasets/upload

GET  /api/datasets/{dataset_id}

GET  /api/datasets/{dataset_id}/data

GET  /api/datasets/{dataset_id}/eda

GET  /api/datasets/{dataset_id}/charts

POST /api/datasets/{dataset_id}/clean

POST /api/ai/ask
```

---

## Project Structure

```text
quicklisis/
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   │   └── analyst.py
│   │   ├── analytics/
│   │   │   └── eda.py
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── data_processing/
│   │   │   ├── profiling.py
│   │   │   └── cleaning.py
│   │   ├── services/
│   │   │   └── dataset_service.py
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── data/
├── docs/
├── .gitignore
└── README.md
```

---

## Environment Variables

### Frontend

```env
VITE_API_URL=https://quicklisis.onrender.com
```

### Backend

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3-flash-preview
```

The Gemini API key is stored only on the backend and is never exposed to the frontend.

---

## Run Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Backend:

```text
http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
```

Create/configure:

```env
VITE_API_URL=http://localhost:8000
```

Run:

```bash
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## Data Analysis Approach

QuickLisis follows a simple principle:

> **Calculate with code. Explain with AI.**

For analytical questions, the backend uses the actual dataframe to perform calculations wherever possible. Gemini is then used to understand the question and explain the calculated result.

This helps reduce incorrect numerical answers and keeps analytical results tied to the uploaded dataset.

---

## Deployment

The current production setup is:

| Component | Platform |
|---|---|
| Frontend | Vercel |
| Backend | Render |
| Source Control | GitHub |
| AI | Google Gemini |

The backend runs with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## Author

**Mohit Kumar Verma**

Built with React, TypeScript, Tailwind CSS, Python, FastAPI, Pandas, NumPy and Google Gemini.

## Links

- **Live Demo:** https://quicklisis.vercel.app
- **Backend:** https://quicklisis.onrender.com
