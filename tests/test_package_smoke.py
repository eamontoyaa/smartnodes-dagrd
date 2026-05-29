from pathlib import Path

from sn_dagrd.inclinometer_io import read_inclinometer_jsonl


def test_sample_data_can_be_read():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "raw" / "santa_rita_1" / "raw_InclinometroSantaRita1.txt"
    df = read_inclinometer_jsonl(path, sensor_type="BNO", drop_duplicate_rows=True)

    assert not df.empty
    assert {"timestamp", "estacion_id", "sensor_id", "roll_deg", "pitch_deg", "yaw_deg"}.issubset(df.columns)
