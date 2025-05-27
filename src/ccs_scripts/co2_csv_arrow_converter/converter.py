# This scripts checks all FMU realizations for missing arrow or csv files
# representing plume extent, area or containment data. If one exists, but
# not the other, it will create the missing file.

import datetime
from enum import StrEnum
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa


def try_convert_csv_to_arrow(
    csv_path: Path,
    arrow_path: Path,
    force_overwrite: bool = False,
) -> bool:
    """
    Convert a CSV file to an Arrow file if the Arrow file does not exist or
    if the force_overwrite flag is set to True.
    """
    if not csv_path.exists():
        return False
    if arrow_path.exists() and not force_overwrite:
        return False

    df = pd.read_csv(csv_path).rename(columns={"date": "DATE"})
    _write_data_frame_to_arrow(df, arrow_path)
    return True


def try_convert_arrow_to_csv(
    csv_path: Path,
    arrow_path: Path,
    force_overwrite: bool = False,
) -> bool:
    """
    Convert an Arrow file to a CSV file if the CSV file does not exist or
    if the force_overwrite flag is set to True.
    """
    if not arrow_path.exists():
        return False
    if csv_path.exists() and not force_overwrite:
        return False

    df = _read_data_frame_from_arrow(arrow_path)
    df.to_csv(csv_path, index=False)
    return True


def try_convert_containment_csv_to_arrow(
    csv_path: Path,
    arrow_path: Path,
    kept_columns: List[str],
    force_overwrite: bool = False,
) -> bool:
    """
    Convert a CSV file to an Arrow file for containment data if the Arrow
    file does not exist or if the force_overwrite flag is set to True.
    The CSV file is expected to have a "date" column and an "amount" column.
    The Arrow file will have a "DATE" column and one column for each unique
    combination of the kept columns, with the sum of the "amount" column
    for each combination.
    """
    if not csv_path.exists():
        return False
    if arrow_path.exists() and not force_overwrite:
        return False

    raw_df = pd.read_csv(csv_path)
    entries = []
    for date, gf0 in raw_df.groupby("date"):
        data = {
            "DATE": date,
        }
        for keys, gf1 in gf0.groupby(kept_columns):
            data["amount--" + "--".join(keys)] = gf1["amount"].sum()
        entries.append(data)

    df = pd.DataFrame(entries)
    _write_data_frame_to_arrow(df, arrow_path)
    return True


def apply_to_realizations(
    root_dir: Path,
    realization_pattern: str,
    kept_columns: List[str],
    overwrite_arrow: bool = False,
    overwrite_csv: bool = False,
) -> None:
    """
    Apply the conversion functions to all realizations matching the
    realization_pattern. The pattern should be a glob pattern relative to
    the current directory. The function will create missing Arrow files
    for CSV files and vice versa. The overwrite_arrow and overwrite_csv
    flags control whether existing files should be overwritten.
    """
    assert not overwrite_csv or not overwrite_arrow
    for realization_dir in root_dir.glob(realization_pattern):
        # Extract paths for CSV and Arrow files
        csv_path_1 = _get_csv_path(realization_dir, _FileType.PLUME_EXTENT)
        csv_path_2 = _get_csv_path(realization_dir, _FileType.PLUME_AREA)
        csv_path_3 = _get_csv_path(realization_dir, _FileType.CONTAINMENT)
        arrow_path_1 = _get_arrow_path(realization_dir, _FileType.PLUME_EXTENT)
        arrow_path_2 = _get_arrow_path(realization_dir, _FileType.PLUME_AREA)
        arrow_path_3 = _get_arrow_path(realization_dir, _FileType.CONTAINMENT)

        conversions = 0
        # Try creating missing Arrow files for all realizations
        conversions += try_convert_csv_to_arrow(csv_path_1, arrow_path_1, overwrite_arrow)
        conversions += try_convert_csv_to_arrow(csv_path_2, arrow_path_2, overwrite_arrow)
        conversions += try_convert_containment_csv_to_arrow(csv_path_3, arrow_path_3, kept_columns, overwrite_arrow)

        # Try creating missing CSV files for all realizations
        conversions += try_convert_arrow_to_csv(csv_path_1, arrow_path_1, overwrite_csv)
        conversions += try_convert_arrow_to_csv(csv_path_2, arrow_path_2, overwrite_csv)
        # No conversion for containment data from Arrow to CSV yet

        print(f"Processed {realization_dir}: {conversions} conversions made.")


class _FileType(StrEnum):
    PLUME_EXTENT = "plume_extent"
    PLUME_AREA = "plume_area"
    CONTAINMENT = "containment"


def _write_data_frame_to_arrow(df: pd.DataFrame, arrow_path: Path) -> None:
    # Extract the timestamp in milliseconds from df. We must use the
    # datetime module, since pandas cannot handle dates outside the range of
    # 1677-09-21 to 2262-04-11. Afterwards we need to explicitly define the
    # schema, since we cannot create a data frame that is directly convertible to
    # Arrow.
    dates = [
        datetime.datetime(*[int(t) for t in d.split("-")])
        for d in df["DATE"]
    ]
    dates_ms = [(d - datetime.datetime(1970, 1, 1)).total_seconds() * 1000 for d in dates]
    non_date_df = df.drop(columns=["DATE"])

    fields = [pa.field("DATE", pa.timestamp('ms'))]
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


def _get_csv_path(realization_dir: Path, file_type: _FileType) -> Path:
    full_path = realization_dir / "share" / "results" / "tables"
    if file_type == _FileType.PLUME_EXTENT:
        full_path = full_path / "plume_extent.csv"
    elif file_type == _FileType.PLUME_AREA:
        full_path = full_path / "plume_area.csv"
    elif file_type == _FileType.CONTAINMENT:
        full_path = full_path / "plume_mass.csv"
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    return full_path


def _get_arrow_path(realization_dir: Path, file_type: _FileType) -> Path:
    return _get_csv_path(realization_dir, file_type).with_suffix(".arrow")


def _read_data_frame_from_arrow(arrow_path: Path) -> pd.DataFrame:
    with pa.ipc.open_file(arrow_path) as f:
        table = f.read_all()
    return table.to_pandas()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create missing Arrow/CSV files for FMU realizations")
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="Root directory for the glob pattern",
        default=Path(".").resolve(),
    )
    parser.add_argument(
        "--realization-pattern",
        type=str,
        help="Glob pattern (relative to current directory) for an FMU realization directory",
        default="realization-*/iter-*",
    )
    parser.add_argument(
        "--kept-columns",
        type=str,
        help="Comma-separated list of columns to keep when converting containment data",
        default="phase,containment",
    )
    parser.add_argument(
        "--force-arrow-overwrite",
        type=bool,
        default=False,
        help="Overwrite existing Arrow files even if they already exist",
    )
    parser.add_argument(
        "--force-csv-overwrite",
        type=bool,
        default=False,
        help="Overwrite existing CSV files even if they already exist",
    )
    args = parser.parse_args()
    apply_to_realizations(
        args.root_dir,
        args.realization_pattern,
        args.kept_columns.split(","),
        args.force_arrow_overwrite,
        args.force_csv_overwrite,
    )


if __name__ == "__main__":
    main()
