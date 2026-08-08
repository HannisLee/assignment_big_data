"""
L11 Assignment - Schema-Aligned Table Recommendation

This script implements a small reproducible example of the recommender design:
1. represent each table by column names and column types;
2. generate only schema-aligned candidates with at least one column match;
3. rank candidates by a hybrid score using schema evidence, query-log evidence,
   personalization, freshness, and quality;
4. generate figures used in report.md.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
TOP_K = 10
MATCH_THRESHOLD = 0.55


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[Column, ...]
    co_usage: int
    popularity: int
    user_affinity: float
    freshness_days: int
    quality: float


QUERY_TABLE = TableInfo(
    name="query_sales_table",
    columns=(
        Column("city", "string"),
        Column("date", "date"),
        Column("sales", "number"),
        Column("product_id", "string"),
    ),
    co_usage=0,
    popularity=0,
    user_affinity=0.0,
    freshness_days=0,
    quality=1.0,
)


CATALOG = [
    TableInfo(
        "daily_city_sales",
        (Column("city", "string"), Column("date", "date"), Column("revenue", "number"), Column("units", "number")),
        co_usage=92,
        popularity=98,
        user_affinity=0.92,
        freshness_days=1,
        quality=0.97,
    ),
    TableInfo(
        "web_ad_clicks",
        (Column("city", "string"), Column("date", "date"), Column("clicks", "number"), Column("campaign_id", "string")),
        co_usage=76,
        popularity=85,
        user_affinity=0.87,
        freshness_days=0,
        quality=0.92,
    ),
    TableInfo(
        "marketing_spend",
        (Column("city", "string"), Column("date", "date"), Column("channel", "string"), Column("spend", "number")),
        co_usage=81,
        popularity=80,
        user_affinity=0.83,
        freshness_days=1,
        quality=0.90,
    ),
    TableInfo(
        "daily_weather",
        (Column("city", "string"), Column("date", "date"), Column("temperature", "number"), Column("rainfall", "number")),
        co_usage=64,
        popularity=73,
        user_affinity=0.78,
        freshness_days=2,
        quality=0.93,
    ),
    TableInfo(
        "product_catalog",
        (Column("product_id", "string"), Column("category", "string"), Column("brand", "string"), Column("price", "number")),
        co_usage=58,
        popularity=88,
        user_affinity=0.80,
        freshness_days=5,
        quality=0.95,
    ),
    TableInfo(
        "returns_by_city",
        (Column("city", "string"), Column("date", "date"), Column("return_amount", "number"), Column("return_count", "number")),
        co_usage=53,
        popularity=70,
        user_affinity=0.69,
        freshness_days=2,
        quality=0.88,
    ),
    TableInfo(
        "inventory_snapshot",
        (Column("date", "date"), Column("product_id", "string"), Column("stock", "number"), Column("warehouse_city", "string")),
        co_usage=47,
        popularity=77,
        user_affinity=0.63,
        freshness_days=0,
        quality=0.91,
    ),
    TableInfo(
        "customer_segments",
        (Column("customer_id", "string"), Column("city", "string"), Column("segment", "string"), Column("lifetime_value", "number")),
        co_usage=39,
        popularity=66,
        user_affinity=0.74,
        freshness_days=6,
        quality=0.86,
    ),
    TableInfo(
        "calendar_events",
        (Column("date", "date"), Column("holiday", "string"), Column("country", "string"), Column("event_type", "string")),
        co_usage=33,
        popularity=61,
        user_affinity=0.58,
        freshness_days=12,
        quality=0.89,
    ),
    TableInfo(
        "city_population",
        (Column("city", "string"), Column("year", "integer"), Column("population", "number"), Column("region", "string")),
        co_usage=28,
        popularity=64,
        user_affinity=0.52,
        freshness_days=30,
        quality=0.91,
    ),
    TableInfo(
        "store_locations",
        (Column("store_id", "string"), Column("city", "string"), Column("region", "string"), Column("opening_date", "date")),
        co_usage=26,
        popularity=60,
        user_affinity=0.49,
        freshness_days=18,
        quality=0.90,
    ),
    TableInfo(
        "exchange_rates",
        (Column("date", "date"), Column("currency", "string"), Column("rate", "number")),
        co_usage=21,
        popularity=57,
        user_affinity=0.46,
        freshness_days=1,
        quality=0.94,
    ),
    TableInfo(
        "support_tickets",
        (Column("ticket_id", "string"), Column("customer_id", "string"), Column("created_at", "datetime"), Column("status", "string")),
        co_usage=11,
        popularity=37,
        user_affinity=0.22,
        freshness_days=3,
        quality=0.82,
    ),
    TableInfo(
        "supplier_contracts",
        (Column("supplier_id", "string"), Column("contract_text", "text"), Column("legal_region", "string")),
        co_usage=4,
        popularity=18,
        user_affinity=0.10,
        freshness_days=45,
        quality=0.74,
    ),
]


SYNONYM_GROUPS = [
    {"city", "store_city", "warehouse_city", "home_city"},
    {"date", "day", "event_date", "opening_date"},
    {"sales", "revenue", "gross_sales", "net_sales", "return_amount", "amount"},
    {"product_id", "sku", "item_id"},
]


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def tokenize(name: str) -> set[str]:
    return {tok for tok in normalize_name(name).split("_") if tok}


def same_synonym_group(a: str, b: str) -> bool:
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    return any(a_norm in group and b_norm in group for group in SYNONYM_GROUPS)


def name_similarity(a: str, b: str) -> float:
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    if a_norm == b_norm:
        return 1.0
    if same_synonym_group(a_norm, b_norm):
        return 0.82

    a_tokens = tokenize(a_norm)
    b_tokens = tokenize(b_norm)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return overlap / union


def type_compatibility(a: str, b: str) -> float:
    a = a.lower()
    b = b.lower()
    numeric = {"number", "integer", "float", "decimal"}
    temporal = {"date", "datetime", "timestamp"}
    string_like = {"string", "text", "categorical"}

    if a == b:
        return 1.0
    if a in numeric and b in numeric:
        return 0.90
    if a in temporal and b in temporal:
        return 0.85
    if a in string_like and b in string_like:
        return 0.70
    return 0.0


def column_match_score(query_col: Column, table_col: Column) -> float:
    name_score = name_similarity(query_col.name, table_col.name)
    type_score = type_compatibility(query_col.dtype, table_col.dtype)
    return 0.70 * name_score + 0.30 * type_score


def schema_alignment(query: TableInfo, table: TableInfo) -> dict:
    alignments = []
    for q_col in query.columns:
        scored = [
            (column_match_score(q_col, t_col), t_col)
            for t_col in table.columns
        ]
        best_score, best_col = max(scored, key=lambda item: item[0])
        if best_score >= MATCH_THRESHOLD:
            alignments.append((q_col, best_col, best_score))

    if not alignments:
        return {"schema_score": 0.0, "alignments": []}

    coverage = len(alignments) / len(query.columns)
    average_strength = sum(score for _, _, score in alignments) / len(alignments)
    max_strength = max(score for _, _, score in alignments)
    schema_score = 0.55 * coverage + 0.35 * average_strength + 0.10 * max_strength
    return {"schema_score": schema_score, "alignments": alignments}


def freshness_score(days: int) -> float:
    return math.exp(-days / 21.0)


def rank_tables(query: TableInfo, catalog: list[TableInfo]) -> list[dict]:
    max_co_usage = max(table.co_usage for table in catalog) or 1
    max_popularity = max(table.popularity for table in catalog) or 1
    ranked = []

    for table in catalog:
        alignment = schema_alignment(query, table)
        if not alignment["alignments"]:
            continue

        schema_score = alignment["schema_score"]
        log_score = 0.70 * (table.co_usage / max_co_usage) + 0.30 * (table.popularity / max_popularity)
        personal_score = table.user_affinity
        fresh_score = freshness_score(table.freshness_days)
        quality_score = table.quality

        final_score = (
            0.50 * schema_score
            + 0.25 * log_score
            + 0.10 * personal_score
            + 0.10 * quality_score
            + 0.05 * fresh_score
        )

        ranked.append(
            {
                "table": table,
                "schema": schema_score,
                "log": log_score,
                "personal": personal_score,
                "quality": quality_score,
                "freshness": fresh_score,
                "final": final_score,
                "alignments": alignment["alignments"],
            }
        )

    return sorted(ranked, key=lambda row: (-row["final"], row["table"].name))


def alignment_text(row: dict) -> str:
    return ", ".join(
        f"{q.name}->{t.name} ({score:.2f})"
        for q, t, score in row["alignments"]
    )


def draw_box(ax, x, y, w, h, title, subtitle, color="#f7fafc", edge="#9fb4c7"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=8.5, color="#586979")


def draw_arrow(ax, x1, y1, x2, y2, color="#6b7c8f"):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
    )
    ax.add_patch(arrow)


def make_pipeline_diagram(out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "Schema-Aligned Table Recommendation Pipeline", fontsize=18, fontweight="bold", color="#233142")
    ax.text(0.03, 0.89, "Offline indexing learns from catalog schemas and query logs; online ranking returns about 10 aligned tables.", fontsize=10.5, color="#586979")

    draw_box(ax, 0.04, 0.66, 0.18, 0.14, "Database tables", "schemas, types, profiles", "#edf7ff", "#4b91d1")
    draw_box(ax, 0.04, 0.42, 0.18, 0.14, "Query log", "sessions, joins, clicks", "#fff4e7", "#e67e22")
    draw_box(ax, 0.28, 0.55, 0.20, 0.18, "Offline builder", "normalize columns\nbuild schema and log indexes", "#f7fafc", "#8aa1b4")
    draw_box(ax, 0.54, 0.66, 0.18, 0.14, "Column index", "name/type/semantic keys", "#eefaf2", "#2ecc71")
    draw_box(ax, 0.54, 0.42, 0.18, 0.14, "Log model", "co-use graph + time decay", "#eefaf2", "#2ecc71")
    draw_box(ax, 0.04, 0.14, 0.18, 0.13, "User query table", "columns only are enough", "#f3efff", "#8e63ce")
    draw_box(ax, 0.30, 0.14, 0.18, 0.13, "Candidate generation", "must have >= 1 match", "#edf7ff", "#4b91d1")
    draw_box(ax, 0.56, 0.14, 0.18, 0.13, "Hybrid ranking", "schema + log + user + quality", "#fff4e7", "#e67e22")
    draw_box(ax, 0.79, 0.14, 0.16, 0.13, "Top-10 result", "tables + matched columns", "#eefaf2", "#2ecc71")

    draw_arrow(ax, 0.22, 0.73, 0.28, 0.65)
    draw_arrow(ax, 0.22, 0.49, 0.28, 0.62)
    draw_arrow(ax, 0.48, 0.64, 0.54, 0.73)
    draw_arrow(ax, 0.48, 0.60, 0.54, 0.49)
    draw_arrow(ax, 0.22, 0.205, 0.30, 0.205)
    draw_arrow(ax, 0.48, 0.205, 0.56, 0.205)
    draw_arrow(ax, 0.74, 0.205, 0.79, 0.205)
    draw_arrow(ax, 0.63, 0.42, 0.39, 0.27)
    draw_arrow(ax, 0.63, 0.66, 0.39, 0.27)

    fig.tight_layout()
    fig.savefig(out_dir / "system_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_score_breakdown(out_dir: Path, ranked: list[dict]):
    top_rows = ranked[:TOP_K]
    labels = [row["table"].name for row in top_rows]
    components = [
        ("Schema", "schema", 0.50, "#3498db"),
        ("Query log", "log", 0.25, "#e67e22"),
        ("Personal", "personal", 0.10, "#9b59b6"),
        ("Quality", "quality", 0.10, "#2ecc71"),
        ("Freshness", "freshness", 0.05, "#95a5a6"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_positions = list(range(len(top_rows)))
    left = [0.0] * len(top_rows)
    for label, key, weight, color in components:
        values = [row[key] * weight for row in top_rows]
        ax.barh(y_positions, values, left=left, color=color, edgecolor="white", label=label)
        left = [l + v for l, v in zip(left, values)]

    for y, row in zip(y_positions, top_rows):
        ax.text(row["final"] + 0.008, y, f"{row['final']:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(row["final"] for row in top_rows) + 0.10)
    ax.set_xlabel("Final weighted score")
    ax.set_title("Top-10 Table Ranking Score Breakdown", fontsize=15, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=True)
    ax.grid(axis="x", color="#edf1f5")
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "score_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_update_loop_diagram(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.92, "Online Update Strategy", fontsize=18, fontweight="bold", color="#233142")
    ax.text(0.04, 0.86, "New tables and new query-log events are incorporated without rebuilding the whole system each time.", fontsize=10.5, color="#586979")

    draw_box(ax, 0.05, 0.60, 0.18, 0.15, "New table upload", "schema parser runs", "#edf7ff", "#4b91d1")
    draw_box(ax, 0.33, 0.60, 0.18, 0.15, "Incremental index", "add column signatures", "#eefaf2", "#2ecc71")
    draw_box(ax, 0.61, 0.60, 0.18, 0.15, "Candidate service", "visible in seconds/minutes", "#fff4e7", "#e67e22")

    draw_box(ax, 0.05, 0.28, 0.18, 0.15, "New query events", "views, joins, selections", "#fff4e7", "#e67e22")
    draw_box(ax, 0.33, 0.28, 0.18, 0.15, "Streaming counters", "time-decayed co-use", "#eefaf2", "#2ecc71")
    draw_box(ax, 0.61, 0.28, 0.18, 0.15, "Periodic retrain", "refresh rank weights", "#f3efff", "#8e63ce")

    draw_box(ax, 0.82, 0.43, 0.14, 0.16, "Ranking API", "fresh top-10 list", "#eefaf2", "#2ecc71")

    draw_arrow(ax, 0.23, 0.675, 0.33, 0.675)
    draw_arrow(ax, 0.51, 0.675, 0.61, 0.675)
    draw_arrow(ax, 0.79, 0.675, 0.84, 0.56)
    draw_arrow(ax, 0.23, 0.355, 0.33, 0.355)
    draw_arrow(ax, 0.51, 0.355, 0.61, 0.355)
    draw_arrow(ax, 0.79, 0.355, 0.84, 0.46)

    ax.text(0.05, 0.12, "Cold start rule: if a table is new and has little log evidence, rank it mainly by schema match, quality, and freshness.", fontsize=10, color="#586979")
    ax.text(0.05, 0.07, "Stability rule: keep old recommendations explainable by showing matched columns and the score components.", fontsize=10, color="#586979")

    fig.tight_layout()
    fig.savefig(out_dir / "online_update_loop.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_ranking(ranked: list[dict]):
    print("=" * 90)
    print("L11 Schema-Aligned Table Recommendation")
    print("=" * 90)
    query_schema = ", ".join(f"{col.name}:{col.dtype}" for col in QUERY_TABLE.columns)
    print(f"Query table: {QUERY_TABLE.name}")
    print(f"Query schema: {query_schema}")
    print()
    print(f"Top-{TOP_K} recommended schema-aligned tables")
    print("-" * 90)
    print(f"{'rank':>4}  {'table':<22} {'final':>7} {'schema':>7} {'log':>7}  matched columns")
    print("-" * 90)
    for idx, row in enumerate(ranked[:TOP_K], start=1):
        print(
            f"{idx:>4}  {row['table'].name:<22} "
            f"{row['final']:>7.3f} {row['schema']:>7.3f} {row['log']:>7.3f}  "
            f"{alignment_text(row)}"
        )

    excluded = [table.name for table in CATALOG if not schema_alignment(QUERY_TABLE, table)["alignments"]]
    print()
    print("Excluded because no column reaches the alignment threshold:")
    print("  " + ", ".join(excluded))


def main():
    ranked = rank_tables(QUERY_TABLE, CATALOG)
    print_ranking(ranked)
    make_pipeline_diagram(OUTPUT_DIR)
    make_score_breakdown(OUTPUT_DIR, ranked)
    make_update_loop_diagram(OUTPUT_DIR)
    print()
    print("Generated figures:")
    for filename in ["system_pipeline.png", "score_breakdown.png", "online_update_loop.png"]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
