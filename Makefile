DB ?= ifrs9.duckdb
ORIG ?= data/raw/sample_orig_*.txt
PERF ?= data/raw/sample_svcg_*.txt
FRED ?= data/raw/fred
LAG ?= 0

.PHONY: install test lint format sql-lint data-synthetic data-freddie quality eda clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

sql-lint:
	sqlfluff lint sql --dialect duckdb

data-synthetic:
	python -m src.data --db $(DB) --synthetic

data-freddie:
	python -m src.data --db $(DB) --orig "$(ORIG)" --perf "$(PERF)" --fred-dir "$(FRED)" --lag-months $(LAG)

quality:
	python -m src.data_quality --db $(DB)

eda:
	python -m src.eda --db $(DB)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml
