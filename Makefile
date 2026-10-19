.PHONY: install test lint format sql-lint clean

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

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml
