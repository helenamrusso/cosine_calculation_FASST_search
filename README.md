# FASST Results - Cosine Re-calculation

This script is intended for calculating the cosine similarities between a Universal Spectrum Identifier (USI) and a spectrum from a local MGF file. This is recommended when FASST searches are done but the user wants to have a higher confidence in the cosine similarities. The FASST searches rely on pre-indexed, pre-filtered spectra, and calculating a 'raw cosine' value of a match that was obtained in FASST would increase the confidence of the match.

## Overview

This tool processes mass spectrometry data by:

1. Downloading spectral data from the GNPS2 metabolomics USI API
2. Calculating cosine similarities between the downloaded spectrum and the spectrum of an MGF (used as input in the FASST searches)
3. Supporting resume functionality with checkpoint system
4. Utilizing multithreading for downloads and multiprocessing for calculations

## Requirements

### Python Dependencies

```bash
pip install polars numpy requests matchms concurrent.futures
```

### System Requirements

- Python 3.7+
- Sufficient RAM for batch processing (configurable batch sizes)
- Internet connection for USI data downloads

## Configuration

### File Paths

Edit the following paths in the script to match your setup:

```python
input_table_path = "/path/to/your/matches.tsv"
query_mgf_path = "/path/to/your/input_mgf.mgf"
output_table_path = "/path/to/output/matches_cosine_output.tsv"
temp_json_dir = "/path/to/temp/storage/"
checkpoint_file = "/path/to/checkpoint.txt"
error_log_file = "/path/to/download_errors.log"
```

### Performance Settings

```python
max_workers = 3        # Concurrent downloads (please respect API rate limits and terms of service)
batch_size = 3000      # USIs per batch (adjust based on available memory)
cosine_workers = 10    # Parallel cosine calculation processes - dependent on cores in your system
```

## Input Format

### Input Table (`matches.tsv`)

Tab-separated file containing at minimum:

- **USI column**: Universal Spectrum Identifiers
- **Scan column**: Scan identifiers matching your MGF file

### Query MGF File

Standard MGF format with spectra containing `scans` metadata that matches the `Scan` column in your input table.

## Usage

### Basic Usage

```bash
python fasst_cosine_calculation_multithread_optimized.py
```

### Resume from Checkpoint

The script automatically detects incomplete processing and resumes from the last checkpoint:

```bash
# Script will automatically identify unprocessed USIs and continue
python fasst_cosine_calculation_multithread_optimized.py
```

## Output

### Output Table

The script generates a TSV file with all original columns plus:

- **cosine_raw**: Cosine similarity scores (float) or 'N/A' for failed calculations

### Log Files

- **Checkpoint file**: Tracks processing progress for resume capability
- **Error log**: Records download failures with timestamps for debugging
