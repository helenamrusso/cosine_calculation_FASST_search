import os
import json
import time
import random
import polars as pl
import numpy as np
import requests
from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy
from datetime import datetime
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from threading import Lock

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("matchms").setLevel(logging.ERROR)

# === Settings ===
input_table_path = "/home/helena/carnitines/FASST_repeat/combined_tables/matches.tsv"
query_mgf_path = "/home/helena/carnitines/FASST_repeat/Carnitine_library_2025_testing.mgf"
output_table_path = "/home/helena/carnitines/FASST_repeat/combined_tables/matches_cosine_calculation_output.tsv"
temp_json_dir = "/home/helena/carnitines/FASST_repeat/temp_json_storage/"
checkpoint_file = "/home/helena/carnitines/FASST_repeat/checkpoint.txt"
error_log_file = "/home/helena/carnitines/FASST_repeat/download_errors.log"

# Threading settings
max_workers = 5  # Number of concurrent downloads
batch_size = 3000  # Number of USIs to process in each batch
cosine_workers = 50  # Number of processes for parallel cosine calculations

# === Initialize ===
os.makedirs(temp_json_dir, exist_ok=True)
file_lock = Lock()  # For thread-safe file operations

def load_checkpoint():
    """Load the last processed USI index from checkpoint file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(usi_index):
    """Save the current USI index to checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        f.write(str(usi_index))

def get_failed_usis(output_df):
    """Get USIs that need to be processed (have null cosine_raw values)"""
    # Filter rows with null cosine_raw and get unique USIs
    failed_rows = output_df.filter(pl.col("cosine_raw").is_null())
    failed_usis = failed_rows.select("USI").unique().to_series().to_list()
    return failed_usis

def download_usi_json(usi):
    """Download JSON for a given USI"""
    json_url = f"https://api.metabolomics-usi.gnps2.org/json/?usi1={usi}"
    
    try:
        start_time = time.time()
        response = requests.get(json_url, timeout=15)
        elapsed = time.time() - start_time
        
        with file_lock:
            print(f"[{usi}] Status {response.status_code} - {elapsed:.2f}s")
        
        if response.status_code == 200:
            return usi, response.json()
        else:
            raise Exception(f"Status code {response.status_code}")
            
    except Exception as e:
        # Log error (thread-safe)
        with file_lock:
            with open(error_log_file, "a") as log:
                log.write(f"{usi}\t{e}\t{datetime.now().isoformat()}\n")
        return usi, None
    
    # Add small delay to avoid hammering server
    time.sleep(random.uniform(0.5, 1.0))

def json_to_spectrum_data(json_data):
    """Convert JSON data to spectrum data (peaks only) for serialization"""
    if json_data is None or "peaks" not in json_data:
        return None
    
    try:
        peaks = np.array(json_data["peaks"])
        return {
            'mz': peaks[:, 0].tolist(),
            'intensities': peaks[:, 1].tolist()
        }
    except Exception:
        return None

def spectrum_data_to_spectrum(spectrum_data, metadata=None):
    """Convert spectrum data back to matchms Spectrum object"""
    if spectrum_data is None:
        return None
    
    try:
        return Spectrum(
            mz=np.array(spectrum_data['mz']),
            intensities=np.array(spectrum_data['intensities']),
            metadata=metadata or {}
        )
    except Exception:
        return None

def calculate_cosine_batch(args):
    """Calculate cosine similarities for a batch of spectrum pairs (for multiprocessing)"""
    reference_spectrum_data, query_spectrum_data_list, tolerance = args
    
    if reference_spectrum_data is None:
        return [None] * len(query_spectrum_data_list)
    
    # Convert reference spectrum data to spectrum
    reference_spectrum = spectrum_data_to_spectrum(reference_spectrum_data)
    if reference_spectrum is None:
        return [None] * len(query_spectrum_data_list)
    
    results = []
    cosine_greedy = CosineGreedy(tolerance=tolerance)
    
    for query_spectrum_data in query_spectrum_data_list:
        if query_spectrum_data is None:
            results.append(None)
            continue
            
        query_spectrum = spectrum_data_to_spectrum(query_spectrum_data)
        if query_spectrum is None:
            results.append(None)
            continue
        
        try:
            score = cosine_greedy.pair(reference_spectrum, query_spectrum)
            results.append(score["score"])
        except Exception:
            results.append(None)
    
    return results

def process_usi_batch(usi_batch, df, output_df, query_spectra):
    """Process a batch of USIs with multithreaded downloads and parallel cosine calculations"""
    
    # Step 1: Download all USIs in the batch concurrently
    print(f"\nDownloading batch of {len(usi_batch)} USIs...")
    usi_json_results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download jobs
        future_to_usi = {executor.submit(download_usi_json, usi): usi for usi in usi_batch}
        
        # Collect results as they complete
        for future in as_completed(future_to_usi):
            usi, json_data = future.result()
            usi_json_results[usi] = json_data
    
    # Step 2: Pre-filter all USIs in this batch at once (MUCH faster)
    print("Preparing cosine calculations...")

    # Single filter operation for all USIs in the batch
    batch_rows = output_df.filter(
        (pl.col("USI").is_in(usi_batch)) & (pl.col("cosine_raw").is_null())
    )

    cosine_tasks = []
    task_metadata = []

    for usi in usi_batch:
        json_data = usi_json_results.get(usi)
        if json_data is None:
            print(f"Skipping USI (download failed): {usi}")
            continue
        
        # Convert JSON to spectrum data for serialization
        reference_spectrum_data = json_to_spectrum_data(json_data)
        if reference_spectrum_data is None:
            print(f"Skipping USI (spectrum creation failed): {usi}")
            continue
        
        # Get rows for this specific USI from pre-filtered batch
        usi_rows = batch_rows.filter(pl.col("USI") == usi)
        
        if len(usi_rows) == 0:
            print(f"USI {usi}: No rows need processing (all already have cosine values)")
            continue
            
        print(f"USI {usi}: Found {len(usi_rows)} rows needing processing")
        
        # Prepare query spectrum data for this USI
        query_spectrum_data_list = []
        scan_ids = []
        
        for row in usi_rows.iter_rows(named=True):
            # FIXED: Keep the original datatype of Scan column
            scan_id = row["Scan"]  # Don't convert to string here
            query_spectrum = query_spectra.get(str(scan_id))  # Convert to string only for lookup
            
            if query_spectrum is None:
                query_spectrum_data_list.append(None)
            else:
                # Convert query spectrum to serializable data
                query_spectrum_data = {
                    'mz': query_spectrum.mz.tolist(),
                    'intensities': query_spectrum.intensities.tolist()
                }
                query_spectrum_data_list.append(query_spectrum_data)
            
            scan_ids.append(scan_id)  # Keep original datatype
        
        if query_spectrum_data_list:
            cosine_tasks.append((reference_spectrum_data, query_spectrum_data_list, 0.2))
            task_metadata.append((usi, scan_ids))
    
    # Step 3: Calculate cosine similarities in parallel
    print(f"Calculating cosine similarities for {len(cosine_tasks)} USI batches...")
    
    if len(cosine_tasks) > 0:
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=min(cosine_workers, len(cosine_tasks))) as executor:
            cosine_results = list(executor.map(calculate_cosine_batch, cosine_tasks))
        elapsed = time.time() - start_time
        total_calculations = sum(len(task[1]) for task in cosine_tasks)
        print(f"Completed {total_calculations} cosine calculations in {elapsed:.2f}s using {min(cosine_workers, len(cosine_tasks))} processes")
    else:
        cosine_results = []
    
    # Step 4: Update dataframe with results (OPTIMIZED)
    print("Updating dataframe...")
    
    # Collect all updates in a single operation instead of individual row updates
    all_updates = []
    
    for task_idx, (usi, scan_ids) in enumerate(task_metadata):
        if task_idx >= len(cosine_results):
            continue
            
        cosine_scores = cosine_results[task_idx]
        
        for scan_idx, scan_id in enumerate(scan_ids):
            if scan_idx >= len(cosine_scores):
                continue
                
            cosine_score = cosine_scores[scan_idx]
            # Convert None to 'N/A' to avoid reprocessing in future runs
            if cosine_score is None:
                cosine_score = 'N/A'
            
            all_updates.append({
                'USI': usi,
                'Scan': scan_id,
                'cosine_score': cosine_score
            })
        
        # Print progress less frequently
        if task_idx % 5 == 0:  # Every 5 USIs instead of every USI
            print(f"  Processed {task_idx + 1}/{len(task_metadata)} USIs")
    
    # Create updates dataframe and apply updates
    if all_updates:
        # FIXED: Infer schema from the data rather than forcing specific types
        # This ensures the datatypes match the original dataframe
        updates_df = pl.DataFrame(all_updates)
        
        # Ensure USI column is string type to match the main dataframe
        updates_df = updates_df.with_columns(
            pl.col('USI').cast(pl.Utf8),
            # Keep Scan column as-is (it should match the original datatype now)
            # cosine_score as string to handle 'N/A' values
            pl.col('cosine_score').cast(pl.Utf8)
        )
        
        # Single join operation to update only the rows that need updating
        print("Applying all updates in single operation...")
        updated_output_df = output_df.join(
            updates_df,
            on=['USI', 'Scan'],
            how='left'
        ).with_columns(
            # Use the new cosine_score if available, otherwise keep the old cosine_raw
            pl.when(pl.col('cosine_score').is_not_null())
            .then(pl.col('cosine_score'))
            .otherwise(pl.col('cosine_raw'))
            .alias('cosine_raw')
        ).drop('cosine_score')  # Remove the temporary column
    else:
        updated_output_df = output_df
    
    print(f"Updated {len(all_updates) if all_updates else 0} rows")
    return updated_output_df

def main():
    print("Loading input table...")
    # Load the input table with Polars - handle mixed data types
    try:
        df = pl.read_csv(
            input_table_path, 
            separator="\t",
            infer_schema_length=10000,
            ignore_errors=True
        )
    except Exception as e:
        print(f"Error reading CSV with default settings: {e}")
        print("Trying with more flexible settings...")
        df = pl.read_csv(
            input_table_path, 
            separator="\t",
            infer_schema_length=0,  # Read all as strings first
            ignore_errors=True
        )
    print(f"Loaded table with {len(df)} rows")
    
    # Check if output file exists and load it
    if os.path.exists(output_table_path):
        print("Loading existing output table...")
        output_df = pl.read_csv(
            output_table_path,
            separator="\t",
            has_header=True,
            infer_schema_length=0,         # scan full file for types
            truncate_ragged_lines=True,    # ignore extra fields on long rows
            ignore_errors=True             # skip type/parse errors rather than fail
        )
        print(f"Existing output has {len(output_df)} rows")
    else:
        # Initialize output dataframe with cosine_raw column
        output_df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("cosine_raw"))
    
    # Load query spectra from MGF file
    print("Loading query MGF file...")
    query_spectra_list = list(load_from_mgf(query_mgf_path))
    query_spectra = {str(s.get("scans")): s for s in query_spectra_list}
    print(f"Loaded {len(query_spectra)} query spectra")
    
    # Get USIs that need processing (failed or missing)
    print("Identifying USIs that need processing...")
    failed_usis = get_failed_usis(output_df)
    print(f"Found {len(failed_usis)} USIs that need processing")
    
    if len(failed_usis) == 0:
        print("All USIs have been successfully processed!")
        return
    
    # Reset checkpoint for failed USIs
    save_checkpoint(0)
    
    # Process failed USIs in batches
    for batch_start in range(0, len(failed_usis), batch_size):
        batch_end = min(batch_start + batch_size, len(failed_usis))
        current_batch = failed_usis[batch_start:batch_end]
        
        print(f"\n=== Processing retry batch {batch_start//batch_size + 1}: USIs {batch_start + 1}-{batch_end} ===")
        
        # Process the batch
        output_df = process_usi_batch(current_batch, df, output_df, query_spectra)
        
        # Save progress after each batch
        print(f"Saving progress after retry batch {batch_start//batch_size + 1}...")
        output_df.write_csv(output_table_path, separator="\t")
        save_checkpoint(batch_end)
    
    # Final save
    print("Saving final results...")
    output_df.write_csv(output_table_path, separator="\t")
    
    # Final summary
    final_failed = get_failed_usis(output_df)
    print(f"Processing complete! {len(final_failed)} USIs still need processing.")
    
    if len(final_failed) > 0:
        print("You can run this script again to retry the remaining failed USIs.")

if __name__ == "__main__":
    main()