# QuickLisis Backend

FastAPI backend for the QuickLisis data-analysis platform.

## Run

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs
