.PHONY: dev test lint eval up down ingest

dev:    ; uvicorn rag_agent.api.main:app --reload
test:   ; pytest -q
lint:   ; ruff check src tests evals && mypy src --ignore-missing-imports
eval:   ; python evals/run_eval.py --url http://localhost:8000
up:     ; docker compose up --build -d
down:   ; docker compose down -v
ingest: ; python scripts/ingest.py --path ./corpus --source $(SOURCE)
