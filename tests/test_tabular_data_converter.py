import io
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pytest import fixture

from ccs_scripts.tabular_data_converter.tabular_data_converter import (
    DataFormat,
    batch_convert_in_directory,
    convert_tabular_data,
    main,
)


@fixture
def mock_data_frame():
    csv_text = """date,all_SGAS,amethyst_SGAS,ruby_SGAS,topaz_SGAS,all_AMFG,amethyst_AMFG,ruby_AMFG,topaz_AMFG,all_XMF2,amethyst_XMF2,ruby_XMF2,topaz_XMF2
2025-01-01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2027-01-01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2031-01-01,1960100.0,1960100.0,1140000.0,1159100.0,8758100.0,8635100.0,8409800.0,8658100.0,8605100.0,7995200.0,7787800.0,8562000.0
2033-01-01,2538900.0,2538900.0,1214900.0,1190100.0,8788100.0,8676100.0,8487900.0,8678100.0,8728100.0,8535000.0,8223700.0,8665000.0
2225-01-01,3708100.0,3588400.0,1207700.0,1231000.0,8809100.0,8747100.0,8547700.0,8760100.0,8620900.0,8346300.0,8208300.0,8528900.0
2250-01-01,3716100.0,3596400.0,1208600.0,1212000.0,8809100.0,8747100.0,8547700.0,8760100.0,8685100.0,8513400.0,8311100.0,8604000.0
2300-01-01,3771100.0,3633400.0,1246700.0,1211000.0,8809100.0,8757100.0,8557700.0,8760100.0,8607800.0,8363400.0,8208400.0,8463900.0
2400-01-01,3864100.0,3773300.0,1244800.0,1201000.0,8809100.0,8757100.0,8557700.0,8770100.0,8627900.0,8423400.0,8190300.0,8501100.0
2450-01-01,3913900.0,3813200.0,1253800.0,1207100.0,8809100.0,8757100.0,8557700.0,8770100.0,8607900.0,8363300.0,8150100.0,8480800.0
2500-01-01,3943800.0,3832200.0,1234800.0,1217000.0,8809100.0,8757100.0,8557700.0,8770100.0,8675900.0,8476400.0,8223100.0,8588800.0
"""  # noqa: E501
    return pd.read_csv(io.StringIO(csv_text))


@fixture
def mock_containment_data_frame():
    return pd.read_csv(
        Path(__file__).parent / "testdata_co2_tables" / "plume_mass.csv",
        parse_dates=["date"],
    )


# =============================================================================
# Basic Conversion Tests
# =============================================================================
# Tests for core conversion functionality between CSV and Arrow formats


def test_convert_csv_to_arrow(mock_data_frame):
    """Test basic CSV to Arrow conversion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"
        mock_data_frame.to_csv(csv_path, index=False)

        # Test conversion
        assert convert_tabular_data(csv_path, arrow_path)
        assert arrow_path.exists()

        # Test that the Arrow file is not empty
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows == len(mock_data_frame)
        assert table.num_columns == len(mock_data_frame.columns)

        # Test that conversion is not done again without overwrite
        assert not convert_tabular_data(csv_path, arrow_path, overwrite=False)

        # ... unless forced
        assert convert_tabular_data(csv_path, arrow_path, overwrite=True)
        # Test that the Arrow file is not empty after forced conversion
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows == len(mock_data_frame)
        assert table.num_columns == len(mock_data_frame.columns)


def test_convert_arrow_to_csv(mock_data_frame):
    """Test basic Arrow to CSV conversion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"

        # Dump data frame to arrow format
        table = pa.Table.from_pandas(mock_data_frame)
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)
        assert arrow_path.exists()

        assert convert_tabular_data(arrow_path, csv_path)
        assert csv_path.exists()

        # Test that the CSV file is not empty
        df = pd.read_csv(csv_path)
        assert not df.empty
        assert df.shape == mock_data_frame.shape

        # Test that conversion is not done again without overwrite
        assert not convert_tabular_data(arrow_path, csv_path, overwrite=False)

        # ... unless forced
        assert convert_tabular_data(arrow_path, csv_path, overwrite=True)
        # Test that the CSV file is not empty after forced conversion
        df = pd.read_csv(csv_path)
        assert not df.empty
        assert df.shape == mock_data_frame.shape


def test_convert_csv_to_arrow_with_aggregation(mock_containment_data_frame):
    """Test CSV to Arrow conversion with date-based aggregation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"
        mock_containment_data_frame.to_csv(csv_path, index=False)

        # Test conversion with aggregation
        assert convert_tabular_data(
            csv_path,
            arrow_path,
            aggregation_columns=["phase", "containment"],
        )
        assert arrow_path.exists()

        # Test that the Arrow file is not empty
        # The data will have transformed and the shape will be different
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows > 0
        # More columns after aggregation due to pivoting
        assert table.num_columns > len(["date", "phase", "containment"])


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================
# Tests for handling missing files and error conditions


def test_skip_if_missing():
    """Test skip_if_missing behavior."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "nonexistent.csv"
        arrow_path = Path(temp_dir) / "output.arrow"

        # Should return False and not raise error when skip_if_missing=True
        result = convert_tabular_data(csv_path, arrow_path, skip_if_missing=True)
        assert result is False
        assert not arrow_path.exists()


def test_explicit_format_specification(mock_data_frame):
    """Test that explicit format specification overrides extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"
        mock_data_frame.to_csv(csv_path, index=False)

        # Convert with explicit format specification
        assert convert_tabular_data(
            csv_path,
            arrow_path,
            src_format=DataFormat.CSV,
            dst_format=DataFormat.ARROW,
        )
        assert arrow_path.exists()


# =============================================================================
# Batch Processing Tests
# =============================================================================
# Tests for batch conversion of multiple files in directories


def test_batch_convert_in_directory(mock_data_frame):
    """Test batch conversion of multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple CSV files
        for i in range(3):
            csv_path = temp_path / f"test_{i}.csv"
            mock_data_frame.to_csv(csv_path, index=False)

        # Batch convert to Arrow
        conversions = batch_convert_in_directory(
            root_dir=temp_path,
            src_pattern="*.csv",
            dst_format="arrow",
        )

        assert conversions == 3

        # Check that Arrow files were created
        for i in range(3):
            arrow_path = temp_path / f"test_{i}.arrow"
            assert arrow_path.exists()


def test_batch_convert_with_suffix(mock_data_frame):
    """Test batch conversion with custom suffix."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a CSV file
        csv_path = temp_path / "test.csv"
        mock_data_frame.to_csv(csv_path, index=False)

        # Batch convert with suffix
        conversions = batch_convert_in_directory(
            root_dir=temp_path,
            src_pattern="*.csv",
            dst_format="arrow",
            dst_suffix="_converted",
        )

        assert conversions == 1

        # Check that Arrow file with suffix was created
        arrow_path = temp_path / "test_converted.arrow"
        assert arrow_path.exists()


def test_custom_date_and_amount_columns():
    """Test conversion with custom date and amount column names."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"

        # Create data with custom column names
        data = pd.DataFrame(
            {
                "timestamp": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
                "region": ["A", "A", "B", "B"],
                "value": [100, 200, 150, 250],
            }
        )
        data.to_csv(csv_path, index=False)

        # Convert with aggregation using custom column names
        assert convert_tabular_data(
            csv_path,
            arrow_path,
            date_column="timestamp",
            aggregation_columns=["region"],
            amount_column="value",
        )

        assert arrow_path.exists()
        table = pa.ipc.open_file(arrow_path).read_all()
        # Should have DATE column and aggregated value columns
        assert table.num_rows == 2  # Two unique timestamps
        assert table.num_columns > 1  # DATE + aggregated columns


# =============================================================================
# CLI Interface Tests
# =============================================================================
# Tests for command-line interface functionality and argument parsing


def test_cli_with_dst_argument(mock_data_frame):
    """Test CLI: tabular_data_converter data.csv --dst data.arrow"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        tables_dir = Path(temp_dir) / "realization-dir" / "share" / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tables_dir / "data.csv"
        arrow_path = tables_dir / "data.arrow"
        mock_data_frame.to_csv(csv_path, index=False)

        # Run CLI
        main([str(csv_path), "--dst", str(arrow_path)])

        # Verify output
        assert arrow_path.exists()
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows == len(mock_data_frame)


def test_cli_with_format_argument(mock_data_frame):
    """Test CLI: tabular_data_converter data.csv --format arrow"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        tables_dir = Path(temp_dir) / "realization-dir" / "share" / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tables_dir / "data.csv"
        arrow_path = tables_dir / "data.arrow"  # Auto-generated
        mock_data_frame.to_csv(csv_path, index=False)

        # Run CLI
        main([str(csv_path), "--format", "arrow"])

        # Verify output
        assert arrow_path.exists()
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows == len(mock_data_frame)


def test_cli_with_aggregation(mock_containment_data_frame):
    """Test CLI: tabular_data_converter data.csv --format arrow --aggregate-columns phase,zone"""  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        tables_dir = Path(temp_dir) / "realization-dir" / "share" / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tables_dir / "data.csv"
        arrow_path = tables_dir / "data.arrow"

        # Use containment data which has phase column
        # Add a zone column for testing
        df = mock_containment_data_frame.copy()
        df["zone"] = "zone1"  # Add zone column
        df.to_csv(csv_path, index=False)

        # Run CLI with aggregation
        main([str(csv_path), "--format", "arrow", "--aggregate-columns", "phase,zone"])

        # Verify output
        assert arrow_path.exists()
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows > 0

        # Check that aggregation happened (should have pivoted columns)
        column_names = table.schema.names
        assert "DATE" in column_names
        # Should have columns like "amount--phase_value--zone_value"
        amount_columns = [col for col in column_names if col.startswith("amount--")]
        assert len(amount_columns) > 0


def test_cli_batch_processing(mock_data_frame):
    """Test CLI: tabular_data_converter --root-dir realization-dir --src-pattern '**/*.csv' --format arrow"""  # noqa: E501

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure with multiple CSV files
        tables_dir = Path(temp_dir) / "realization-dir" / "share" / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple CSV files
        csv_files = ["data1.csv", "data2.csv", "data3.csv"]
        for csv_file in csv_files:
            csv_path = tables_dir / csv_file
            mock_data_frame.to_csv(csv_path, index=False)

        # Run CLI in batch mode
        root_dir = Path(temp_dir) / "realization-dir"
        main(
            [
                "--root-dir",
                str(root_dir),
                "--src-pattern",
                "**/*.csv",
                "--format",
                "arrow",
            ]
        )

        # Verify all Arrow files were created
        for csv_file in csv_files:
            arrow_file = csv_file.replace(".csv", ".arrow")
            arrow_path = tables_dir / arrow_file
            assert arrow_path.exists(), f"Expected {arrow_path} to exist"

            # Verify content
            table = pa.ipc.open_file(arrow_path).read_all()
            assert table.num_rows == len(mock_data_frame)


def test_cli_dry_run(mock_data_frame, capsys):
    """Test CLI: tabular_data_converter data.csv --format arrow --dry-run"""

    with tempfile.TemporaryDirectory() as temp_dir:
        tables_dir = Path(temp_dir) / "realization-dir" / "share" / "results" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        csv_path = tables_dir / "data.csv"
        arrow_path = tables_dir / "data.arrow"
        mock_data_frame.to_csv(csv_path, index=False)

        # Run CLI with dry-run
        main([str(csv_path), "--format", "arrow", "--dry-run"])

        # Verify NO output file was created
        assert not arrow_path.exists()

        # Check console output shows dry run message
        captured = capsys.readouterr()
        assert "DRY RUN:" in captured.out
        assert str(csv_path) in captured.out
        assert str(arrow_path) in captured.out
