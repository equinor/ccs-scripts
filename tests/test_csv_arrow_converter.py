import io
import pyarrow as pa
import pandas as pd
import tempfile
from pathlib import Path

from pytest import fixture
from ccs_scripts.csv_arrow_converter.converter import (
    try_convert_arrow_to_csv,
    try_convert_csv_to_arrow,
    try_convert_containment_csv_to_arrow,
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
    """
    return pd.read_csv(io.StringIO(csv_text))


@fixture
def mock_containment_data_frame():
    return pd.read_csv(Path(__file__).parent / "testdata_co2_tables" / "plume_mass.csv", parse_dates=["date"])


def test_convert_csv_to_arrow(mock_data_frame):
    # Dump mock data frame to CSV in a temporary file in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"
        mock_data_frame.to_csv(csv_path, index=False)

        # Test conversion
        assert try_convert_csv_to_arrow(csv_path, arrow_path)
        assert arrow_path.exists()

        # Test that the Arrow file is not empty
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows == len(mock_data_frame)
        assert table.num_columns == len(mock_data_frame.columns)

        # Test that conversion is not done again...
        assert not try_convert_csv_to_arrow(csv_path, arrow_path)

        # ... unless forced
        assert try_convert_csv_to_arrow(csv_path, arrow_path, force_overwrite=True)
        # Test that the Arrow file is not empty after forced conversion
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows == len(mock_data_frame)
        assert table.num_columns == len(mock_data_frame.columns)


def test_convert_arrow_to_csv(mock_data_frame):
    # Dump mock data frame to Arrow in a temporary file in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"

        # Dump data frame to arrow format
        table = pa.Table.from_pandas(mock_data_frame)
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)
        assert arrow_path.exists()
        assert try_convert_arrow_to_csv(csv_path, arrow_path)
        assert csv_path.exists()
        
        # Test that the CSV file is not empty
        df = pd.read_csv(csv_path)
        assert not df.empty
        assert df.shape == mock_data_frame.shape

        # Test that conversion is not done again...
        assert not try_convert_arrow_to_csv(csv_path, arrow_path)

        # ... unless forced
        assert try_convert_arrow_to_csv(csv_path, arrow_path, force_overwrite=True)
        # Test that the CSV file is not empty after forced conversion
        df = pd.read_csv(csv_path)
        assert not df.empty
        assert df.shape == mock_data_frame.shape


def test_convert_containment_csv_to_arrow(mock_containment_data_frame):
    # Dump mock data frame to CSV in a temporary file in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "test.csv"
        arrow_path = Path(temp_dir) / "test.arrow"
        mock_containment_data_frame.to_csv(csv_path, index=False)

        # Test conversion
        assert try_convert_containment_csv_to_arrow(
            csv_path,
            arrow_path,
            kept_columns=["phase", "containment"],
        )
        assert arrow_path.exists()

        # Test that the Arrow file is not empty
        # The data will have transformed and the shape will be different
        table = pa.ipc.open_file(arrow_path).read_all()
        assert table.num_rows > 0
        assert table.num_columns > mock_containment_data_frame.shape[1]
