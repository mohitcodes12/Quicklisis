# QuickLisis

**From raw data to actionable insight.**

QuickLisis is a full-stack AI-powered data analysis platform. The MVP accepts CSV/XLSX datasets, profiles their structure, checks data quality, generates EDA/chart recommendations, previews data, and answers common natural-language analytical questions with deterministic Pandas calculations.

## Stack
- Frontend: React + TypeScript + Vite + Recharts + Tailwind-inspired custom CSS
- Backend: Python + FastAPI + Pandas + NumPy
- File support: CSV/XLSX
- AI architecture: verified calculation first, optional LLM explanation layer
- Current persistence: in-memory dataset registry + local uploaded files

## Run
See `backend/README.md` and `frontend/README.md`.

Sample data: `data/sample/sales.csv`
