from __future__ import annotations

import time
import webbrowser
from pathlib import Path
from threading import RLock
from typing import Any, List

import duckdb
import polars as pl

from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger
from selectron.util.slugify_url import slugify_url

logger = get_logger(__name__)

DB_FILENAME = "selectron.duckdb"

_db_lock = RLock()


def get_db_path() -> Path:
    """Returns the path to the DuckDB database file in the app directory."""
    app_dir = get_app_dir()
    return app_dir / DB_FILENAME


def get_db_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Gets a DuckDB connection, using the default path if none provided."""
    if db_path is None:
        db_path = get_db_path()

    # convert str to Path for uniform handling
    if isinstance(db_path, str):
        db_path = Path(db_path)

    try:
        # Ensure the parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(db_path))
        logger.debug("DuckDB connection successful.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DuckDB at {db_path}: {e}", exc_info=True)
        raise  # re-raise the exception after logging


def setup_db(conn: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    """
    Ensures the database is set up (currently just returns a connection).
    Placeholder for future table creation logic.
    """
    if conn:
        # maybe validate connection?
        logger.debug("Using existing DuckDB connection.")
        return conn
    else:
        # get a new connection if one wasn't provided
        return get_db_connection()


def launch_duckdb_ui(db_path: str) -> duckdb.DuckDBPyConnection | None:
    """Launch the DuckDB web UI and RETURN the connection used.

    Returns None if launching failed.
    """
    logger.info(f"Launching DuckDB UI for database: {db_path}")
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        # Get connection (don't use 'with' here)
        conn = get_db_connection(Path(db_path))
        if not conn:
            # get_db_connection already logs error, just return
            return None

        logger.debug("Installing/loading DuckDB UI extension...")
        conn.execute("INSTALL ui;")
        conn.execute("LOAD ui;")
        logger.debug("Starting UI server...")
        conn.execute("CALL start_ui_server();")
        time.sleep(1)  # Give server a moment to start
        ui_url = "http://localhost:4213"
        logger.info(f"Opening DuckDB UI at {ui_url}")
        webbrowser.open(ui_url)
        print("DuckDB UI server running in background.")
        return conn  # Return the active connection

    except Exception as e:
        logger.error(f"Failed to launch DuckDB UI: {e}", exc_info=True)
        if conn:
            try:
                conn.close()  # Attempt to close connection on error
            except Exception as close_err:
                logger.error(f"Error closing connection after UI launch failure: {close_err}")
        return None


# ---------------------------------------------------------------------------
# Dynamic result-table helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True if the given table exists in the current connection."""
    try:
        res = conn.execute(
            """SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1""",
            [table_name],
        ).fetchone()
        return res is not None
    except Exception as e:
        logger.error(f"NOTE: Failed table_exists check for '{table_name}': {e}")
        return False


def _get_table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Return a list of column names for the given table (empty list if not found)."""
    try:
        # Query information_schema to avoid issues with dots in identifiers
        rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            """,
            [table_name],
        ).fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"NOTE: Failed to fetch columns for table '{table_name}': {e}")
        return []


def _register_temp_df(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> str:
    """Register the given Polars DataFrame as a temporary DuckDB view and return its name."""
    view_name = f"tmp_insert_{int(time.time() * 1000)}"
    try:
        conn.register(view_name, df.to_arrow())  # DuckDB can scan arrow table
    except Exception as e:
        logger.error(f"NOTE: Failed registering temp df '{view_name}': {e}", exc_info=True)
        raise
    return view_name


def _quote_ident(name: str) -> str:
    """Safely quote an identifier (table/column) for DuckDB."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _create_table_from_df(
    conn: duckdb.DuckDBPyConnection, table_name: str, df: pl.DataFrame
) -> None:
    """Create a new DuckDB table using the schema inferred from df."""
    temp_view = _register_temp_df(conn, df)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_quote_ident(table_name)} AS SELECT * FROM {temp_view}"
    )
    conn.unregister(temp_view)


def _insert_df_into_table(
    conn: duckdb.DuckDBPyConnection, table_name: str, df: pl.DataFrame
) -> None:
    """Insert rows from df into existing table (column order is aligned)."""
    existing_cols = _get_table_columns(conn, table_name)
    if not existing_cols:
        logger.warning(
            f"NOTE: Attempting to insert into '{table_name}' but failed to retrieve columns. Skipping insert."
        )
        return

    # Ensure df has all existing columns; fill missing ones with None
    for col in existing_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    # Reorder df columns to match table order
    df = df.select(existing_cols)
    temp_view = _register_temp_df(conn, df)
    conn.execute(f"INSERT INTO {_quote_ident(table_name)} SELECT * FROM {temp_view}")
    conn.unregister(temp_view)


def _generate_incremented_name(base: str, existing_names: set[str]) -> str:
    """Return an incremented table name that doesn't collide with existing_names."""
    suffix = 2
    while True:
        candidate = f"{base}__{suffix}"
        if candidate not in existing_names:
            return candidate
        suffix += 1


def save_parsed_results(url: str, rows: List[dict[str, Any]]) -> None:
    """Save parser extracted rows into a DuckDB table named by the URL slug.

    The function will dynamically create the table if it doesn't exist. If it exists
    but the column schema differs from the incoming rows, a new table with an
    incremented suffix (e.g., slug__2) is created.
    """

    if not rows:
        logger.debug("No rows to save; skipping DuckDB insert.")
        return

    slug = slugify_url(url)
    df = pl.from_dicts(rows)  # type inference via Polars/Arrow

    db_path = get_db_path()

    with _db_lock:  # Serialize DuckDB writes to avoid write-write conflicts
        with get_db_connection(db_path) as conn:
            try:
                # Determine if table exists
                table_name = slug
                if _table_exists(conn, table_name):
                    # Check schema compatibility
                    existing_cols = set(_get_table_columns(conn, table_name))
                    incoming_cols = set(df.columns)
                    missing_cols = incoming_cols - existing_cols
                    if missing_cols:
                        for col in missing_cols:
                            # Default to VARCHAR if we can't map type easily
                            try:
                                conn.execute(
                                    f"ALTER TABLE {_quote_ident(table_name)} ADD COLUMN {_quote_ident(col)} VARCHAR"
                                )
                                existing_cols.add(col)
                            except Exception as e:
                                logger.error(
                                    f"Failed to add column '{col}' to table '{table_name}': {e}",
                                    exc_info=True,
                                )

                    logger.debug(f"Inserting into table '{table_name}'.")
                    _insert_df_into_table(conn, table_name, df)
                else:
                    logger.info(f"Creating new table '{table_name}' for URL '{url}'.")
                    _create_table_from_df(conn, table_name, df)
            except Exception as e:
                logger.error(f"Failed to save parsed results for URL '{url}': {e}", exc_info=True)


def delete_all_tables() -> None:
    """Deletes all user tables in the main schema of the default database."""
    db_path = get_db_path()
    logger.info(f"Attempting to delete all tables from database: {db_path}")
    with _db_lock:
        with get_db_connection(db_path) as conn:
            try:
                # Get all table names in the main schema
                tables = conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()

                if not tables:
                    logger.info("No tables found to delete.")
                    return

                table_count = 0
                for (table_name,) in tables:
                    try:
                        logger.debug(f"Dropping table: {table_name}")
                        conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(table_name)}")
                        table_count += 1
                    except Exception as e:
                        logger.error(f"Failed to drop table '{table_name}': {e}", exc_info=True)
                logger.info(f"Successfully dropped {table_count} tables.")

            except Exception as e:
                logger.error(f"Failed to retrieve or delete tables: {e}", exc_info=True)
