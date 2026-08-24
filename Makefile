.DEFAULT_GOAL := help

# uv run inherits the shell env, which on this machine includes a ROS PYTHONPATH
# that crashes pytest's plugin autoloading. Same guard as semantic-highlighting-slm.
PY := env -u PYTHONPATH uv run

PORT ?= 8000
SOURCE ?=

.PHONY: help install test serve worker seed generate ingest book discover doctor reset-db

help:
	@echo "Setup:"
	@echo "  make install          install/sync dependencies"
	@echo "  make test             run the test suite (no GPU, no weights, no network)"
	@echo "  make seed             insert sample articles so the site is reviewable"
	@echo ""
	@echo "Running:"
	@echo "  make serve            start the reading site on :$(PORT)  (loads no models)"
	@echo "  make worker           start the offline worker: drains the job queue on a schedule"
	@echo ""
	@echo "Pipeline (one-shot, bypasses the queue):"
	@echo "  make doctor           check Ollama, the GPU and the configured models are reachable"
	@echo "  make ingest           queue every new PDF in data/pdfs/ (add RUN=1 to generate now)"
	@echo "  make book SOURCE=x.pdf   split a book by chapter/subchapter into a series (DRY=1, LIMIT=n)"
	@echo "  make generate SOURCE=path/to.pdf   ingest + generate + illustrate + highlight one PDF"
	@echo "  make discover         find papers from what the corpus cites, tags and summarises"
	@echo "                          STRATEGY=references|keywords|prompted   DRY=1   RUN=1"
	@echo ""
	@echo "  make reset-db         delete the SQLite database (articles and queue)"

install:
	uv sync

test:
	$(PY) pytest -q

seed:
	$(PY) python scripts/seed.py

serve:
	$(PY) uvicorn articliser.web.app:app --host 0.0.0.0 --port $(PORT) --reload

worker:
	$(PY) python -m articliser.worker.scheduler

doctor:
	$(PY) python scripts/doctor.py

generate:
	@test -n "$(SOURCE)" || { echo "usage: make generate SOURCE=path/to.pdf"; exit 2; }
	$(PY) python scripts/generate.py "$(SOURCE)"

book:
	@test -n "$(SOURCE)" || { echo "usage: make book SOURCE=path/to/book.pdf [DRY=1] [LIMIT=n]"; exit 2; }
	$(PY) python scripts/book.py "$(SOURCE)" $(if $(DRY),--dry-run,) $(if $(LIMIT),--limit $(LIMIT),)

# RUN=1 generates immediately; without it the PDFs just join the queue.
ingest:
	$(PY) python scripts/ingest.py $(if $(RUN),--run,)

discover:
	$(PY) python scripts/discover.py $(if $(STRATEGY),--strategy $(STRATEGY),) $(if $(DRY),--dry-run,) $(if $(RUN),--run,)

reset-db:
	rm -f data/articliser.db data/articliser.db-wal data/articliser.db-shm
	@echo "database removed; run 'make seed' to repopulate"
