DB ?= ifrs9.duckdb
ORIG ?= data/raw/sample_orig_*.txt,data/raw/orig_*.txt
PERF ?= data/raw/sample_svcg_*.txt,data/raw/perf_*.txt
FRED ?= data/raw/fred
LAG ?= 0

.PHONY: install test lint format sql-lint data-synthetic data-freddie quality eda pd lifetime ecl monitor report check sim clean

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

pd:
	python -m src.pd_pipeline --db $(DB)

lifetime:
	python -m src.lifetime_pipeline --db $(DB)

ecl:
	python -m src.ecl_pipeline --db $(DB)

monitor:
	python -m src.monitoring_pipeline --db $(DB)

report:
	python -m src.report

check: lint test data-synthetic quality
	@echo "All checks passed"

sim:
	python -m src.simulate --outdir data/sim
	python -m src.data --db $(DB) --orig "data/sim/orig_*.txt" --perf "data/sim/perf_*.txt" --fred-dir data/sim/fred

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml
