from pathlib import Path
import yaml

# Project root directory
root = Path().resolve().parent

def load_config(path=root / "configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
    
config = load_config()

### Raw data
# Raw data directory
raw_data_dir =  root.parents[1]  / config['raw_data']['dir']

# Paths to raw data files
rmh_raw_data_path = [
    raw_data_dir / config['raw_data']['rmh_data_file_1'],
    raw_data_dir / config['raw_data']['rmh_data_file_2']
]
lvrh_raw_data_path = raw_data_dir / config['raw_data']['lvrh_data_file']

### Datasets of triage notes
# Data (interim, processed, predictied)
interim_data_dir = root / config['data']['interim_data']
proc_data_dir = root / config['data']['proc_data']
pred_data_dir = root / config['data']['pred_data']

### Ancillaries
# Data to support spelling correction
spell_corr_dir = root / config['data']['spell_corr']

### Resources
# Resources directory
resources_dir = root.parents[2] / config['resources']['dir']

# Dictionary of English words
scowl_dir = resources_dir / config['resources']['scowl_dir']

# Paths to AMT reference lists
amt_ed_path = resources_dir / config['resources']['amt_ed_file']
amt_mp_path = resources_dir / config['resources']['amt_mp_file']
amt_tp_path = resources_dir / config['resources']['amt_tp_file']

# Path to list of Vic suburbs
vic_path = resources_dir / config['resources']['vic_file']

# Path to the list of abbreviations
abbr_path = resources_dir / config['resources']['abbr_file']

# Models directory
models_dir = root / config['models_dir']

# Results directory
results_dir = root / config['results_dir']

### Global variables
N_SPLITS = 5

### Labels
SH_LABELS = [0, 1]

