# DATA-PIPELINE-DEVEPLOPMENT

This repository contains a Python ETL pipeline example built for Colab or local Python execution. The pipeline demonstrates:
- data loading
- inspection and cleaning
- transformation and feature engineering
- export of processed output

## Files
- `ETL_Pipeline.ipynb` — Jupyter notebook for Colab, with Google Drive support and step-by-step ETL workflow.
- `etl_pipeline.py` — Python script that generates sample data, cleans and transforms it, then exports the processed output.
- `requirements.txt` — dependency list for the ETL pipeline.
- `.gitignore` — ignores generated caches and exported CSVs.

## Requirements
- Python 3.10 or newer
- `pandas`
- `numpy`
- `scikit-learn`

## Install
Run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Run the script
Execute the pipeline with:

```bash
python3 etl_pipeline.py
```

The script will create `cleaned_sales_train.csv` and `cleaned_sales_test.csv` in the current directory.

## Colab usage
Open `ETL_Pipeline.ipynb` in Google Colab to run the notebook interactively. The notebook includes Google Drive mounting and sample dataset loading logic.

## Notes
- The Python script automatically installs missing dependencies if needed.
- Generated output CSV files are ignored by `.gitignore` so only source files are tracked in the repository.