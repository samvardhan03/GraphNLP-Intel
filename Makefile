.PHONY: dev test lint docker-up docker-down

dev:
	uvicorn graphnlp.api.app:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check graphnlp/ && mypy graphnlp/

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down
