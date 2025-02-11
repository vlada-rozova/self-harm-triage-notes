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

# Resources directory
resources_dir = root.parents[2] / "Resources"

# Dictionary of English words
scowl_dir = resources_dir / "scowl-2020.12.07/final"

# Paths to AMT reference lists
amt_ed_path = resources_dir / "AMT/Australian-emergency-department-reference-set-20230831-expansion.tsv"
amt_mp_path = resources_dir / "AMT/Medicinal-product-reference-set-20230831-expansion.tsv"
amt_tp_path = resources_dir / "AMT/Trade-product-reference-set-20230831-expansion.tsv"

# Path to the list of abbreviations
abbr_path = resources_dir / "Abbreviations/Clinical Abbreviations List - SESLHDPR 282 cleaned.xlsx"

# Models directory
models_dir = root / "models"

# Results directory
results_dir = root / "results"

# Global variables
N_SPLITS = 5

