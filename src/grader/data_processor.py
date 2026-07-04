"""data_processor.py - the Databricks pipeline engine (Marvel-style).

Handles creating the Unity Catalog infrastructure (catalog, schema, tables,
volume) and saving graded data as Delta tables. Like Marvel's DataProcessor,
it contains NO hardcoded catalog/schema names - everything comes from
ProjectConfig, so the same code targets dev/acc/prd by changing the env.
"""

try:
    from loguru import logger
except ImportError:  # loguru not installed -> use standard logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("grader")

from grader.project_config import ProjectConfig


class DataProcessor:
    """Creates infrastructure and writes exam-grader data to Unity Catalog."""

    def __init__(self, config: ProjectConfig, spark):
        self.config = config
        self.spark = spark

    # -----------------------------------------------------------------
    # 1. INFRASTRUCTURE: create catalog, schema, tables, volume
    # -----------------------------------------------------------------
    def setup_infrastructure(self, create_catalog: bool = False) -> None:
        """Create the schema, five Delta tables, and the Volume.

        By default the CATALOG is assumed to already exist (created in the
        Databricks UI, Marvel-style). Pass create_catalog=True to also create
        it from code.
        """
        base = self.config.base_path
        logger.info(f"Setting up infrastructure at {base}")

        # Catalog (optional - usually created manually in the UI first)
        if create_catalog:
            self.spark.sql(
                f"CREATE CATALOG IF NOT EXISTS {self.config.catalog_name}"
            )
        # Schema (created under the existing catalog)
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {base}")

        # The five Delta tables (mirror the local SQLite store)
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {base}.questions (
                id STRING, type STRING, text STRING, correct_answer STRING,
                rubric STRING, tolerance DOUBLE, max_marks DOUBLE
            ) USING DELTA
        """)
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {base}.answers (
                id STRING, question_id STRING, student_id STRING, answer_text STRING
            ) USING DELTA
        """)
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {base}.grades (
                id STRING, answer_id STRING, question_id STRING, student_id STRING,
                marks_awarded DOUBLE, max_marks DOUBLE, justification STRING,
                confidence DOUBLE, grading_method STRING, status STRING
            ) USING DELTA
        """)
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {base}.audit (
                id STRING, grade_record_id STRING, timestamp STRING, actor STRING,
                action STRING, old_marks DOUBLE, new_marks DOUBLE, reason STRING
            ) USING DELTA
        """)
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {base}.predictions (
                id STRING, created_at STRING, student_id STRING, question_id STRING,
                image_path STRING, answer_read STRING, marks_awarded DOUBLE,
                max_marks DOUBLE, grading_method STRING, model_used STRING
            ) USING DELTA
        """)

        # The Volume for uploaded images
        self.spark.sql(
            f"CREATE VOLUME IF NOT EXISTS {base}.{self.config.volume_name}"
        )
        logger.info("Infrastructure ready: 5 tables + volume.")

    # -----------------------------------------------------------------
    # 2. ENABLE CHANGE DATA FEED (Marvel-style auditing)
    # -----------------------------------------------------------------
    def enable_change_data_feed(self) -> None:
        """Turn on Delta Change Data Feed so table changes are tracked."""
        base = self.config.base_path
        for table in self.config.tables:
            full = f"{base}.{table}"
            self.spark.sql(
                f"ALTER TABLE {full} "
                f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            )
            logger.info(f"Change Data Feed enabled: {full}")

    # -----------------------------------------------------------------
    # 3. SAVE DATA: write graded rows to the Delta tables
    # -----------------------------------------------------------------
    def save_questions(self, questions: list[dict]) -> None:
        """Append question dicts to the questions Delta table."""
        self._append("questions", questions)

    def save_grades(self, grades: list[dict]) -> None:
        """Append grade dicts to the grades Delta table."""
        self._append("grades", grades)

    def save_predictions(self, predictions: list[dict]) -> None:
        """Append prediction dicts (the dataset) to the predictions table."""
        self._append("predictions", predictions)

    def _append(self, table: str, rows: list[dict]) -> None:
        """Convert dicts to a Spark DataFrame and append to a Delta table.

        We reuse the TARGET TABLE's schema so Spark never has to guess column
        types from the data. Guessing fails when a column is all None (e.g. an
        MCQ question has rubric=None), raising CANNOT_DETERMINE_TYPE. Using the
        table's own schema avoids that entirely.
        """
        if not rows:
            logger.info(f"No rows to write to {table}.")
            return
        base = self.config.base_path
        full = f"{base}.{table}"

        # Read the existing table's schema (column names + types).
        target_schema = self.spark.table(full).schema
        cols = [f.name for f in target_schema]

        # Align each row to the table's columns (fill missing keys with None,
        # drop any extra keys), preserving column order.
        aligned = [{c: row.get(c) for c in cols} for row in rows]

        # Build the DataFrame WITH the table's schema so types are explicit.
        sdf = self.spark.createDataFrame(aligned, schema=target_schema)
        sdf.write.format("delta").mode("append").saveAsTable(full)
        logger.info(f"Wrote {len(rows)} row(s) to {full}.")

    # -----------------------------------------------------------------
    # 4. READ helpers (for inspection)
    # -----------------------------------------------------------------
    def show_counts(self) -> None:
        """Print row counts for every table."""
        base = self.config.base_path
        for table in self.config.tables:
            n = self.spark.sql(f"SELECT count(*) AS n FROM {base}.{table}").collect()[0]["n"]
            logger.info(f"{table}: {n} rows")
