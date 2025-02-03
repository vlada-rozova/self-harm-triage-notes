# Self-harm in triage notes

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Detection of self-harm presentations from emergency department triage notes

## Project Organization

```
├── .env               <- Stores environment variables
├── .gitignore         <- Lists files that should be excluded from version control
├── pyproject.toml     <- Project configuration file with package metadata for trial_project and configuration for tools like black
├── environment.yml    <- The requirements file for reproducing the analysis environment
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
│ 
├── data
│   ├── interim       		<- Metadata for collected datasets
│   ├── predicted     		<- Data processed with a final classifier with predcitions appended
│   ├── processed      		<- The final, canonical data sets for modelling
│   └── spelling correction     <- Vocabularies for spelling correction
│
├── models             <- Trained and serialised models, and parameters
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── results            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── tests              <- Tests for custom functions
│
└── self_harm_triage_notes      <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes self_harm_triage_notes a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── text.py                 <- Code to parse free text
    │
    └── dev.py                  <- Code to train an ML model
```

--------

