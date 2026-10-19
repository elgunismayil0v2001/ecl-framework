"""Build the DuckDB database by running the SQL scripts in order."""
from pathlib import Path
import duckdb

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
DB_PATH = Path(__file__).resolve().parents[1] / "ifrs9.duckdb"


def connect(path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path))


def run_sql(con: duckdb.DuckDBPyConnection, filename: str) -> None:
    con.execute((SQL_DIR / filename).read_text())


def build(upto: str = "03_features.sql") -> None:
    """Run SQL scripts up to and including `upto` (04+ need model outputs)."""
    con = connect()
    for f in sorted(SQL_DIR.glob("*.sql")):
        run_sql(con, f.name)
        if f.name == upto:
            break
    con.close()


if __name__ == "__main__":
    build()
