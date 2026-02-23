"""Tests unitaires – pipeline d'ingestion."""
import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import add_rul, validate_dataframe, load_raw_file
from src.config import COLUMN_NAMES, RUL_CLIP


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n_units: int = 3, cycles_per_unit: int = 10) -> pd.DataFrame:
    """Créer un DataFrame minimal C-MAPSS pour les tests."""
    rows = []
    for uid in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row = {
                "unit_id": uid,
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
            }
            # Capteurs avec des valeurs fictives
            for s in range(1, 22):
                row[f"sensor_{s}"] = float(np.random.rand())
            rows.append(row)
    return pd.DataFrame(rows)


# ── Tests ingestion ───────────────────────────────────────────────────────────

class TestLoadRawFile:
    def test_load_raw_file_creates_correct_columns(self, tmp_path):
        """load_raw_file doit produire exactement les colonnes COLUMN_NAMES."""
        # Créer un fichier espace-séparé minimal (26 colonnes)
        data = " ".join(["1"] * 26)
        raw_file = tmp_path / "test.txt"
        raw_file.write_text(f"{data}\n{data}\n")

        df = load_raw_file(raw_file)
        assert list(df.columns) == COLUMN_NAMES

    def test_load_raw_file_dtypes(self, tmp_path):
        """unit_id et cycle doivent être des entiers."""
        data = " ".join(["1"] * 26)
        raw_file = tmp_path / "test.txt"
        raw_file.write_text(f"{data}\n")

        df = load_raw_file(raw_file)
        assert df["unit_id"].dtype == int
        assert df["cycle"].dtype == int


class TestAddRUL:
    def test_add_rul_basic(self):
        """RUL = max_cycle - cycle pour chaque moteur."""
        df = _make_df(n_units=2, cycles_per_unit=10)
        df_rul = add_rul(df, clip=125)

        # Dernier cycle d'un moteur → RUL = 0
        last_rows = df_rul.groupby("unit_id").tail(1)
        assert (last_rows["rul"] == 0).all()

        # Premier cycle → RUL = max_cycle - 1 (clipé si > 125)
        first_rows = df_rul.groupby("unit_id").head(1)
        assert (first_rows["rul"] == min(9, RUL_CLIP)).all()

    def test_add_rul_clipping(self):
        """RUL ne doit pas dépasser RUL_CLIP."""
        df = _make_df(n_units=1, cycles_per_unit=200)
        df_rul = add_rul(df, clip=125)
        assert df_rul["rul"].max() <= 125

    def test_add_rul_non_negative(self):
        """RUL doit toujours être >= 0."""
        df = _make_df(n_units=3, cycles_per_unit=50)
        df_rul = add_rul(df)
        assert (df_rul["rul"] >= 0).all()


class TestValidateDataframe:
    def test_validate_passes_clean_data(self):
        """validate_dataframe ne lève pas d'exception sur des données propres."""
        df = _make_df(n_units=3, cycles_per_unit=10)
        df = add_rul(df)
        report = validate_dataframe(df)

        assert report["n_rows"] == 30
        assert report["n_units"] == 3
        assert report["missing_pct"] == 0.0
        assert report["duplicate_rows"] == 0
        assert report["cycles_monotonic"] is True

    def test_validate_detects_duplicates(self):
        """validate_dataframe doit détecter les doublons."""
        df = _make_df(n_units=1, cycles_per_unit=5)
        df = add_rul(df)
        df_dup = pd.concat([df, df.iloc[:1]], ignore_index=True)  # ajouter un doublon

        with pytest.raises(AssertionError, match="doublon"):
            validate_dataframe(df_dup)

    def test_validate_detects_missing(self):
        """validate_dataframe doit détecter trop de valeurs manquantes."""
        df = _make_df(n_units=1, cycles_per_unit=10)
        df = add_rul(df)
        # Injecter 50% de NaN
        df.loc[df.index[:5], "sensor_1"] = np.nan
        df.loc[df.index[:5], "sensor_2"] = np.nan

        with pytest.raises(AssertionError, match="manquantes"):
            validate_dataframe(df)
