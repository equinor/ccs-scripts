# General-purpose tabular data format converter
# Converts between CSV and Arrow formats with optional date-based aggregation.
# Supports flexible input/output specification and batch processing.

import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa


class DataFormat(Enum):
    """Supported data formats."""

    CSV = "csv"
    ARROW = "arrow"

    @classmethod
    def from_extension(cls, ext: str) -> "DataFormat":
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        if ext == "csv":
            return cls.CSV
        elif ext in ("arrow", "ipc"):
            return cls.ARROW
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @classmethod
    def from_path(cls, path: Path) -> "DataFormat":
        """Get format from file path extension."""
        return cls.from_extension(path.suffix)


def convert_tabular_data(
    src_path: Path,
    dst_path: Path,
    src_format: Optional[DataFormat] = None,
    dst_format: Optional[DataFormat] = None,
    date_column: str = "date",
    aggregation_columns: Optional[List[str]] = None,
    amount_column: str = "amount",
    overwrite: bool = False,
    skip_if_missing: bool = True,
) -> bool:
    """
    Convert tabular data between different formats.

    Args:
        src_path: Path to source file
        dst_path: Path to destination file
        src_format: Source format (auto-detected from extension if None)
        dst_format: Destination format (auto-detected from extension if None)
        date_column: Name of the date column for aggregation
        aggregation_columns: Columns to group by for aggregation (enables aggregation)
        amount_column: Column to sum when aggregating
        overwrite: Whether to overwrite existing destination file
        skip_if_missing: Whether to silently skip if source file doesn't exist

    Returns:
        True if conversion was performed, False otherwise
    """
    # Check if source exists
    if not src_path.exists():
        if skip_if_missing:
            return False
        else:
            raise FileNotFoundError(f"Source file not found: {src_path}")

    # Check if destination exists and overwrite policy
    if dst_path.exists() and not overwrite:
        return False

    # Auto-detect formats if not specified
    if src_format is None:
        src_format = DataFormat.from_path(src_path)
    if dst_format is None:
        dst_format = DataFormat.from_path(dst_path)

    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data based on source format
    if src_format == DataFormat.CSV:
        df = pd.read_csv(src_path)
    elif src_format == DataFormat.ARROW:
        df = _read_data_frame_from_arrow(src_path)
    else:
        raise ValueError(f"Unsupported source format: {src_format}")

    # Apply aggregation if requested
    if aggregation_columns is not None:
        df = _aggregate_by_date(df, date_column, aggregation_columns, amount_column)

    # Normalize date column name for Arrow format
    if dst_format == DataFormat.ARROW and date_column in df.columns:
        df = df.rename(columns={date_column: "DATE"})

    # Save data based on destination format
    if dst_format == DataFormat.CSV:
        df.to_csv(dst_path, index=False)
    elif dst_format == DataFormat.ARROW:
        _write_data_frame_to_arrow(df, dst_path)
    else:
        raise ValueError(f"Unsupported destination format: {dst_format}")

    return True


def _aggregate_by_date(
    df: pd.DataFrame,
    date_column: str,
    aggregation_columns: List[str],
    amount_column: str,
) -> pd.DataFrame:
    """
    Aggregate data by date and specified columns.

    Groups data by date and aggregation columns, summing the amount column
    for each combination. Creates new columns with names like 'amount--key1--key2'.

    Args:
        df: Source DataFrame
        date_column: Name of the date column
        aggregation_columns: Columns to group by
        amount_column: Column to sum

    Returns:
        Aggregated DataFrame
    """
    entries = []
    for date, date_group in df.groupby(date_column):
        data = {date_column: date}

        for keys, group in date_group.groupby(aggregation_columns):
            # Create column name from grouping keys
            if isinstance(keys, tuple):
                column_name = f"{amount_column}--" + "--".join(str(k) for k in keys)
            else:
                column_name = f"{amount_column}--{keys}"
            data[column_name] = group[amount_column].sum()

        entries.append(data)

    return pd.DataFrame(entries)


def batch_convert_in_directory(
    root_dir: Path,
    src_pattern: str,
    dst_format: Union[str, DataFormat],
    dst_suffix: Optional[str] = None,
    date_column: str = "date",
    aggregation_columns: Optional[List[str]] = None,
    amount_column: str = "amount",
    overwrite: bool = False,
    skip_if_missing: bool = True,
) -> int:
    """
    Batch convert files matching a pattern in a directory.

    Args:
        root_dir: Root directory to search for files
        src_pattern: Glob pattern for source files (e.g., "**/*.csv")
        dst_format: Destination format ("csv" or "arrow")
        dst_suffix: Optional suffix to add to destination filename
        date_column: Name of date column for aggregation
        aggregation_columns: Columns to group by (enables aggregation if provided)
        amount_column: Column to sum when aggregating
        overwrite: Whether to overwrite existing files
        skip_if_missing: Whether to skip missing files

    Returns:
        Number of files converted
    """
    if isinstance(dst_format, str):
        dst_format = DataFormat(dst_format.lower())

    dst_ext = "." + dst_format.value
    conversions = 0

    for src_path in root_dir.glob(src_pattern):
        if src_path.is_file():
            # Generate destination path
            if dst_suffix:
                dst_name = src_path.stem + dst_suffix + dst_ext
            else:
                dst_name = src_path.stem + dst_ext
            dst_path = src_path.parent / dst_name

            if convert_tabular_data(
                src_path=src_path,
                dst_path=dst_path,
                date_column=date_column,
                aggregation_columns=aggregation_columns,
                amount_column=amount_column,
                overwrite=overwrite,
                skip_if_missing=skip_if_missing,
            ):
                conversions += 1
                print(f"Converted: {src_path} -> {dst_path}")

    return conversions


def _write_data_frame_to_arrow(df: pd.DataFrame, arrow_path: Path) -> None:
    # Extract the timestamp in milliseconds from df. We must use the
    # datetime module, since pandas cannot handle dates outside the range of
    # 1677-09-21 to 2262-04-11. Afterwards we need to explicitly define the
    # schema, since we cannot create a data frame that is directly convertible to
    # Arrow.
    dates = [
        datetime.datetime(*[int(t) for t in d.split("-")])  # type: ignore[arg-type]
        for d in df["DATE"]
    ]
    dates_ms = [
        (d - datetime.datetime(1970, 1, 1)).total_seconds() * 1000 for d in dates
    ]
    non_date_df = df.drop(columns=["DATE"])

    fields = [pa.field("DATE", pa.timestamp("ms"))]
    for col in non_date_df.columns:
        # Use default type inference for other columns
        fields.append(pa.field(col, pa.infer_type(non_date_df[col])))

    arrays = [dates_ms]
    for col in non_date_df.columns:
        arrays.append(pa.array(non_date_df[col]))

    schema = pa.schema(fields)

    # Create the table with the specified schema
    table = pa.Table.from_arrays(arrays, schema=schema)
    with pa.ipc.new_file(arrow_path, schema) as writer:
        writer.write_table(table)


def _read_data_frame_from_arrow(arrow_path: Path) -> pd.DataFrame:
    with pa.ipc.open_file(arrow_path) as f:
        table = f.read_all()
    return table.to_pandas()


def main(argv=None):
    """Main entry point for CLI.

    Args:
        argv: List of command-line arguments (for testing). If None, uses sys.argv.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert tabular data between CSV and Arrow formats with optional "
            "aggregation."
        ),
        epilog="Examples:\n"  # noqa: E501
        "  tabular_data_converter realization-dir/share/results/tables/data.csv --dst realization-dir/share/results/tables/data.arrow\n"  # noqa: E501
        "  tabular_data_converter realization-dir/share/results/tables/data.csv --format arrow\n"  # noqa: E501
        "  tabular_data_converter realization-dir/share/results/tables/data.csv --format arrow --aggregate-columns phase,zone\n"  # noqa: E501
        "  tabular_data_converter --root-dir realization-dir --src-pattern '**/*.csv' --format arrow",  # noqa: E501
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main input/output arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "src",
        nargs="?",
        type=Path,
        help="Source file to convert",
    )
    group.add_argument(
        "--root-dir",
        type=Path,
        help="Root directory for batch processing",
        metavar="<ROOT_DIR>",
    )

    parser.add_argument(
        "--dst",
        type=Path,
        help="Destination file (format inferred from extension)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "arrow"],
        help="Output format (required if --dst not provided)",
    )
    parser.add_argument(
        "--src-format",
        choices=["csv", "arrow"],
        help="Source format (auto-detected if not specified)",
    )

    # Batch processing options
    parser.add_argument(
        "--src-pattern",
        default="**/*.csv",
        help="Glob pattern for source files in batch mode (default: **/*.csv)",
    )
    parser.add_argument(
        "--dst-suffix",
        help="Suffix to add to destination filenames in batch mode",
    )

    # Data processing options
    parser.add_argument(
        "--date-column",
        default="date",
        help="Name of the date column (default: date)",
    )
    parser.add_argument(
        "--aggregate-columns",
        help=(
            "Comma-separated list of columns to group by for aggregation (enables "
            "aggregation)"
        ),
    )
    parser.add_argument(
        "--amount-column",
        default="amount",
        help=(
            "Name of the amount/value column to sum during aggregation (default: "
            "amount)"
        ),
    )

    # Control options
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files",
    )
    parser.add_argument(
        "--no-skip-missing",
        action="store_true",
        help="Fail if source files are missing (default: skip missing files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without performing actual conversions",
    )

    args = parser.parse_args(argv)

    # Validate arguments
    if args.src and not args.root_dir:
        # Single file mode
        if not args.dst and not args.format:
            parser.error("Either --dst or --format must be provided")

        # Generate destination path if only format provided
        if not args.dst:
            args.dst = args.src.with_suffix(f".{args.format}")

        if not args.src.exists() and not args.no_skip_missing:
            print(f"Source file not found: {args.src}")
            return

    elif args.root_dir:
        # Batch mode
        if not args.format:
            parser.error("--format is required in batch mode")

        if not args.root_dir.exists():
            parser.error(f"Root directory does not exist: {args.root_dir}")

    # Parse aggregation columns
    aggregation_columns = None
    if args.aggregate_columns:
        aggregation_columns = [
            col.strip() for col in args.aggregate_columns.split(",") if col.strip()
        ]
        if not aggregation_columns:
            parser.error(
                "At least one column must be specified for --aggregate-columns"
            )

    # Dry run output
    if args.dry_run:
        if args.src:
            print(f"DRY RUN: Would convert {args.src} -> {args.dst}")
            if aggregation_columns:
                print(f"DRY RUN: Would aggregate by columns: {aggregation_columns}")
        else:
            print(f"DRY RUN: Would batch convert in {args.root_dir}")
            print(f"DRY RUN: Pattern: {args.src_pattern}")
            print(f"DRY RUN: Output format: {args.format}")
            if aggregation_columns:
                print(f"DRY RUN: Would aggregate by columns: {aggregation_columns}")
        return

    try:
        if args.src:
            # Single file conversion
            src_format = DataFormat(args.src_format) if args.src_format else None
            dst_format = DataFormat(args.format) if args.format else None

            success = convert_tabular_data(
                src_path=args.src,
                dst_path=args.dst,
                src_format=src_format,
                dst_format=dst_format,
                date_column=args.date_column,
                aggregation_columns=aggregation_columns,
                amount_column=args.amount_column,
                overwrite=args.overwrite,
                skip_if_missing=not args.no_skip_missing,
            )

            if success:
                print(f"Converted: {args.src} -> {args.dst}")
            else:
                print(f"No conversion needed or file already exists: {args.src}")

        else:
            # Batch conversion
            conversions = batch_convert_in_directory(
                root_dir=args.root_dir,
                src_pattern=args.src_pattern,
                dst_format=args.format,
                dst_suffix=args.dst_suffix,
                date_column=args.date_column,
                aggregation_columns=aggregation_columns,
                amount_column=args.amount_column,
                overwrite=args.overwrite,
                skip_if_missing=not args.no_skip_missing,
            )

            print(f"Batch processing completed: {conversions} files converted")

    except Exception as e:
        parser.error(f"Conversion failed: {e}")


if __name__ == "__main__":
    main()
