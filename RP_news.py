
"""
RavenPack News Preprocessing

Purpose:
    Transform RavenPack news observations into a
    consolidated security-event dataset containing
    53 baseline machine learning predictors.

Inputs:
    Original RavenPack CSV
    Standardized category mapping

Outputs:
    Processed RavenPack Parquet dataset
    Baseline feature list
    Security reference mapping
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


# --------------------------------------------------
# 1. Define the feature structure
# --------------------------------------------------

TARGET_CATEGORIES = [
    "earnings_beat",
    "earnings_miss",
    "revenue_beat",
    "revenue_miss",
    "earnings_guidance",
    "dividend_changes",
    "share_repurchases",
    "equity_issuance",
    "debt_issuance",
    "credit_rating_changes",
    "default_bankruptcy",
    "mergers_acquisitions",
    "asset_purchases_sales",
    "analyst_recommendation_changes",
    "analyst_price_target_changes",
    "management_changes",
    "insider_transactions",
    "lawsuits",
    "regulatory_actions",
    "product_launches",
    "product_recalls",
    "partnerships_contracts"
]

NUMERICAL_FEATURES = [
    "relevance",
    "event_sentiment_score",
    "event_relevance",
    "event_similarity_days"
]

FACT_FEATURES = [
    "is_fact",
    "is_forecast",
    "is_opinion"
]

NOVELTY_FEATURES = [
    "novel_1day",
    "novel_7days"
]

INTERACTION_FEATURES = [
    f"{category}_sentiment"
    for category in TARGET_CATEGORIES
]

BASELINE_FEATURES = (
    TARGET_CATEGORIES +
    NUMERICAL_FEATURES +
    FACT_FEATURES +
    NOVELTY_FEATURES +
    INTERACTION_FEATURES
)

IDENTIFIER_COLUMNS = [
    "rp_entity_id",
    "cusip",
    "rp_story_id",
    "rp_story_event_index",
    "timestamp_utc",
    "category",
    "model_category",
    "event_similarity_key"
]

SUPPORTING_FEATURES = [
    "fact_level",
    "news_type",
    "rp_source_id",
    "source_name",
    "css",
    "nip",
    "peq",
    "bee",
    "bmq",
    "bam",
    "bca",
    "ber",
    "anl_chg",
    "mcq"
]

DISCRETE_SIGNALS = [
    "peq", "bee", "bmq", "bam",
    "bca", "ber", "anl_chg", "mcq"
]

EVENT_KEY = [
    "rp_story_id",
    "cusip",
    "rp_entity_id",
    "rp_story_event_index",
    "category",
    "timestamp_utc"
]

SOURCE_COLUMNS = (
    IDENTIFIER_COLUMNS +
    NUMERICAL_FEATURES +
    SUPPORTING_FEATURES
)

FINAL_COLUMNS = (
    IDENTIFIER_COLUMNS +
    BASELINE_FEATURES +
    SUPPORTING_FEATURES
)


# --------------------------------------------------
# 2. Extract mapped news observations
# --------------------------------------------------

def extract_mapped_news(
    file_path,
    mapping_path,
    output_path,
    chunk_size=100_000
):

    mapping_df = pd.read_parquet(mapping_path)

    assert mapping_df["category"].is_unique

    category_mapping = (
        mapping_df
        .set_index("category")["model_category"]
        .to_dict()
    )

    original_columns = pd.read_csv(
        file_path,
        nrows=0
    ).columns.tolist()

    assert set(SOURCE_COLUMNS) - {"model_category"} <= set(
        original_columns
    )

    assert {"ticker", "isin", "entity_name"} <= set(
        original_columns
    )

    schema = pa.schema([
        (col, pa.string())
        for col in original_columns + ["model_category"]
    ])

    temporary_path = output_path.with_suffix(".tmp.parquet")

    rows_processed = 0
    rows_mapped = 0

    with pq.ParquetWriter(
        temporary_path,
        schema,
        compression="snappy"
    ) as writer:

        for i, chunk in enumerate(
            pd.read_csv(
                file_path,
                chunksize=chunk_size,
                dtype="string"
            ),
            start=1
        ):

            rows_processed += len(chunk)

            chunk["model_category"] = (
                chunk["category"].map(category_mapping)
            )

            mapped_chunk = chunk.loc[
                chunk["model_category"].isin(
                    TARGET_CATEGORIES
                )
            ]

            rows_mapped += len(mapped_chunk)

            if not mapped_chunk.empty:

                writer.write_table(
                    pa.Table.from_pandas(
                        mapped_chunk,
                        schema=schema,
                        preserve_index=False
                    )
                )

            if i % 50 == 0:

                print(
                    f"Processed: {rows_processed:,} | "
                    f"Mapped: {rows_mapped:,}"
                )

    if rows_mapped == 0:
        raise ValueError("No mapped news records found.")

    temporary_path.replace(output_path)

    print(f"Original observations: {rows_processed:,}")
    print(f"Mapped observations: {rows_mapped:,}")

    return rows_mapped


# --------------------------------------------------
# 3. Consolidate redundant event records
# --------------------------------------------------

def consolidate_news(
    mapped_path,
    consolidated_path,
    reference_path
):

    temp_folder = (
        consolidated_path.parent / "duckdb_temp"
    )

    temp_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    con = duckdb.connect(":memory:")

    try:

        con.execute("SET memory_limit = '6GB'")
        con.execute("SET threads = 2")

        con.execute(
            "SET temp_directory = ?",
            [str(temp_folder.resolve())]
        )

        con.execute(
            "SET preserve_insertion_order = false"
        )

        con.read_parquet(
            str(mapped_path.resolve())
        ).create_view("mapped_news")

        original_count = con.execute("""
            SELECT COUNT(*)
            FROM mapped_news
        """).fetchone()[0]

        # Preserve security reference combinations

        reference_df = con.execute("""
            SELECT DISTINCT
                rp_entity_id,
                entity_name,
                cusip,
                ticker,
                isin
            FROM mapped_news
        """).df()

        reference_df.to_parquet(
            reference_path,
            index=False
        )

        # Preserve all remaining news information

        all_columns = con.execute("""
            DESCRIBE SELECT * FROM mapped_news
        """).df()["column_name"].tolist()

        retained_columns = [
            col for col in all_columns
            if col not in ["ticker", "isin"]
        ]

        retained_sql = ", ".join(
            '"' + col.replace('"', '""') + '"'
            for col in retained_columns
        )

        key_sql = ", ".join(EVENT_KEY)

        temporary_path = consolidated_path.with_suffix(
            ".tmp.parquet"
        )

        temporary_path.unlink(missing_ok=True)

        sql_path = str(
            temporary_path.resolve()
        ).replace("'", "''")

        con.execute(f"""
            COPY (
                SELECT DISTINCT {retained_sql}
                FROM mapped_news
            )
            TO '{sql_path}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY
            )
        """)

        con.read_parquet(
            str(temporary_path.resolve())
        ).create_view("consolidated_news")

        consolidated_count = con.execute("""
            SELECT COUNT(*)
            FROM consolidated_news
        """).fetchone()[0]

        unique_event_count = con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT {key_sql}
                FROM consolidated_news
            )
        """).fetchone()[0]

        # Stop if conflicting event records remain

        if unique_event_count != consolidated_count:

            raise ValueError(
                "Conflicting security-event records "
                "remain after consolidation."
            )

        # Verify complete retained information

        missing_records = con.execute(f"""
            SELECT COUNT(*)
            FROM (
                SELECT {retained_sql}
                FROM mapped_news

                EXCEPT

                SELECT {retained_sql}
                FROM consolidated_news
            )
        """).fetchone()[0]

        if missing_records != 0:

            raise ValueError(
                "Distinct news information was lost."
            )

    finally:

        con.close()

    temporary_path.replace(consolidated_path)

    print(
        f"Consolidated observations: "
        f"{consolidated_count:,}"
    )

    print(
        f"Redundant observations removed: "
        f"{original_count - consolidated_count:,}"
    )

    return consolidated_count


# --------------------------------------------------
# 4. Construct baseline news predictors
# --------------------------------------------------

def construct_features(consolidated_path):

    df = pd.read_parquet(
        consolidated_path,
        columns=SOURCE_COLUMNS
    )

    # Convert numerical data

    for col in NUMERICAL_FEATURES + ["css", "nip"]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("float64")

    for col in DISCRETE_SIGNALS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("Int8")

    # Preserve UTC timestamps

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        format="mixed",
        errors="coerce",
        utc=True
    )

    # Preserve security identifiers as strings

    df["cusip"] = df["cusip"].astype("string")

    # Convert categorical information

    for col in [
        "category",
        "model_category",
        "fact_level",
        "news_type",
        "source_name"
    ]:

        df[col] = df[col].astype("category")

    # Validate source variables

    assert df[EVENT_KEY].notna().all().all()

    assert not df[EVENT_KEY].astype(
        "string"
    ).apply(
        lambda col: col.str.strip().eq("")
    ).any().any()

    assert df["model_category"].isin(
        TARGET_CATEGORIES
    ).all()

    assert df["fact_level"].isin(
        ["fact", "forecast", "opinion"]
    ).all()

    assert df["relevance"].between(0, 100).all()

    assert df["event_relevance"].between(0, 100).all()

    assert df["event_sentiment_score"].between(
        -1, 1
    ).all()

    assert df["event_similarity_days"].between(
        0, 365
    ).all()

    # News category indicators

    for category in TARGET_CATEGORIES:

        df[category] = (
            df["model_category"] == category
        ).astype("int8")

    # Fact-level indicators

    for value in ["fact", "forecast", "opinion"]:

        df[f"is_{value}"] = (
            df["fact_level"] == value
        ).astype("int8")

    # Novelty indicators

    df["novel_1day"] = (
        df["event_similarity_days"] >= 1
    ).astype("int8")

    df["novel_7days"] = (
        df["event_similarity_days"] >= 7
    ).astype("int8")

    # Category-specific sentiment interactions

    for category in TARGET_CATEGORIES:

        df[f"{category}_sentiment"] = np.where(
            df[category] == 1,
            df["event_sentiment_score"],
            0.0
        ).astype("float32")

    # Validate the final feature structure

    assert len(BASELINE_FEATURES) == 53

    assert len(set(BASELINE_FEATURES)) == 53

    assert len(FINAL_COLUMNS) == 75

    assert len(set(FINAL_COLUMNS)) == 75

    assert df[BASELINE_FEATURES].notna().all().all()

    assert all(
        np.isfinite(
            df[col].to_numpy(dtype=float)
        ).all()
        for col in BASELINE_FEATURES
    )

    assert (
        df[TARGET_CATEGORIES].sum(axis=1) == 1
    ).all()

    assert (
        df[FACT_FEATURES].sum(axis=1) == 1
    ).all()

    assert (
        df["novel_7days"] <= df["novel_1day"]
    ).all()

    assert np.allclose(
        df[INTERACTION_FEATURES].sum(axis=1),
        df["event_sentiment_score"],
        atol=1e-7
    )

    assert not df.duplicated(
        subset=EVENT_KEY
    ).any()

    # Organize the final dataset

    df = df[FINAL_COLUMNS]

    df = df.sort_values([
        "timestamp_utc",
        "cusip",
        "rp_story_id",
        "rp_story_event_index"
    ]).reset_index(drop=True)

    return df


# --------------------------------------------------
# 5. Run the preprocessing pipeline
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="RavenPack news preprocessing"
    )

    parser.add_argument(
        "--input",
        default="News Ml Project Data .csv"
    )

    parser.add_argument(
        "--mapping",
        default="data/ravenpack_category_mapping.parquet"
    )

    parser.add_argument(
        "--output-dir",
        default="data"
    )

    parser.add_argument(
        "--from-consolidated",
        default=None
    )

    args = parser.parse_args()

    output_folder = Path(args.output_dir)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.from_consolidated:

        consolidated_path = Path(
            args.from_consolidated
        )

    else:

        mapped_path = (
            output_folder /
            "ravenpack_full_mapped.parquet"
        )

        consolidated_path = (
            output_folder /
            "ravenpack_consolidated.parquet"
        )

        reference_path = (
            output_folder /
            "ravenpack_security_reference.parquet"
        )

        extract_mapped_news(
            Path(args.input),
            Path(args.mapping),
            mapped_path
        )

        consolidate_news(
            mapped_path,
            consolidated_path,
            reference_path
        )

    df = construct_features(consolidated_path)

    final_path = (
        output_folder /
        "ravenpack_processed.parquet"
    )

    temporary_path = final_path.with_suffix(
        ".tmp.parquet"
    )

    df.to_parquet(
        temporary_path,
        index=False,
        compression="snappy"
    )

    metadata = pq.ParquetFile(temporary_path)

    assert metadata.metadata.num_rows == len(df)

    assert metadata.schema_arrow.names == FINAL_COLUMNS

    temporary_path.replace(final_path)

    pd.DataFrame({
        "feature": BASELINE_FEATURES
    }).to_csv(
        output_folder /
        "ravenpack_baseline_features.csv",
        index=False
    )

    print("\nPreprocessing completed successfully.")

    print(f"Observations: {len(df):,}")
    print(f"Variables: {len(df.columns)}")
    print(f"Baseline predictors: {len(BASELINE_FEATURES)}")

    print(
        "Constant baseline predictors:",
        [
            col for col in BASELINE_FEATURES
            if df[col].nunique() <= 1
        ]
    )

    print(f"Output: {final_path}")


if __name__ == "__main__":

    main()
