# Common dev/ops targets

.PHONY: help build up down logs test unit smoke fmt lint k6

help:
	@echo "Targets: build up down logs test unit smoke fmt lint k6"

build:
	docker compose -f docker-compose.prod.yml --env-file env.example build

up:
	docker compose -f docker-compose.prod.yml --env-file env.example up -d

down:
	docker compose -f docker-compose.prod.yml down -v

logs:
	docker compose -f docker-compose.prod.yml logs -f --tail=100

unit:
	RUN_UNIT=1 pytest -q

smoke:
	@echo "Run CI-like smoke locally (requires jq)"
	docker compose -f docker-compose.prod.yml --env-file env.example up -d postgres redis ingestion agent nlu rag
	bash -c 'for i in {1..30}; do curl -sf http://localhost:8001/health && break || sleep 2; done'
	bash -c 'for i in {1..30}; do curl -sf http://localhost:8005/health && break || sleep 2; done'
	bash -c 'for i in {1..30}; do curl -sf http://localhost:8003/health && break || sleep 2; done'
	bash -c 'for i in {1..30}; do curl -sf http://localhost:8004/health && break || sleep 2; done'
	curl -sf http://localhost:8001/metrics | head -n 10
	curl -sf http://localhost:8005/metrics | head -n 10
	docker compose -f docker-compose.prod.yml down -v

fmt:
	ruff format

lint:
	ruff check .

k6:
	docker run --rm -i grafana/k6 run - < scripts/load/k6-smoke.js
