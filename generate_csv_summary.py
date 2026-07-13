#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_\n"
    return df.to_markdown(index=False) + "\n"


def summarize_csv(csv_path: Path, output_md: Path) -> None:
    df = pd.read_csv(csv_path)

    rows, cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2) if len(df) else missing

    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(exclude="number")

    column_profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": df.notna().sum().values,
            "nulls": missing.values,
            "null_%": missing_pct.values,
            "unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )

    top_missing = (
        column_profile.sort_values(["nulls", "null_%"], ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    numeric_stats = pd.DataFrame()
    if not num_df.empty:
        numeric_stats = (
            num_df.describe(percentiles=[0.25, 0.5, 0.75])
            .T.reset_index()
            .rename(columns={"index": "column"})
        )
        # Keep a compact set
        keep = [c for c in ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"] if c in numeric_stats.columns]
        numeric_stats = numeric_stats[keep].round(4)

    categorical_sections = []
    if not cat_df.empty:
        for c in cat_df.columns[:20]:  # cap for readability
            vc = df[c].astype("object").fillna("<NA>").value_counts(dropna=False).head(10)
            part = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
            part["pct"] = (part["count"] / len(df) * 100).round(2) if len(df) else 0
            categorical_sections.append((c, part))

    dup_count = int(df.duplicated().sum())

    md = []
    md.append(f"# CSV Summary: `{csv_path.name}`\n")
    md.append("## Dataset Overview\n")
    md.append(f"- **Rows:** {rows}")
    md.append(f"- **Columns:** {cols}")
    md.append(f"- **Duplicate rows:** {dup_count}\n")

    md.append("## Column Profile\n")
    md.append(md_table(column_profile))

    md.append("## Missing Values (Top 15)\n")
    md.append(md_table(top_missing))

    md.append("## Numeric Summary\n")
    if numeric_stats.empty:
        md.append("_No numeric columns found._\n")
    else:
        md.append(md_table(numeric_stats))

    md.append("## Categorical Top Values\n")
    if not categorical_sections:
        md.append("_No categorical columns found._\n")
    else:
        for col, table in categorical_sections:
            md.append(f"### `{col}`\n")
            md.append(md_table(table))

    output_md.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 generate_csv_summary.py <input.csv> [output.md]")
        return 1

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    output_md = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) > 2
        else csv_path.with_name("summary.md")
    )

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    summarize_csv(csv_path, output_md)
    print(f"Created: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())