# QuickLisis — Windows Run Guide

## Backend

Open PowerShell in the `backend` folder:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Keep this terminal open.

## Frontend

Open a second PowerShell in the `frontend` folder:

```powershell
npm install
copy .env.example .env
npm run dev
```

Open the Vite URL shown in the terminal, normally:

`http://localhost:5173`

## Production build test

```powershell
npm run build
```

The build should create `frontend/dist`.
