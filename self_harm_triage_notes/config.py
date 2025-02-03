from pathlib import Path

# Project root directory
root = Path().resolve().parent

# Raw data directory
raw_data_dir = root.parents[1] / "Data/RMH LV May23"

# Datasets of triage notes
data_interim_dir = root / "data/interim"
data_proc_dir = root / "data/processed"
data_pred_dir = root / "data/predicted"

# Data to support spelling correction
spell_corr_dir = root / "data/spelling correction"