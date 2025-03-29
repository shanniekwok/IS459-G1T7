# WORKING PYTHON SCRIPT TO COMBINE hhblock_dataset

import pandas as pd
import numpy as np
import os

# Define the path to the data folder
data_folder = '../../Data/hhblock_dataset/'

def process_block(block_num):
    """Process a single block file if it hasn't been processed already."""
    input_file = f'block_{block_num}.csv'
    output_file = f'block_{block_num}_processed.csv'
    
    input_path = os.path.join(data_folder, input_file)
    output_path = os.path.join(data_folder, output_file)
    
    # Check if output file already exists
    if os.path.exists(output_path):
        print(f"✔ {output_file} already exists. Skipping processing.")
        return True
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"✘ Input file {input_file} not found. Skipping.")
        return False
    
    # Load the CSV file
    df = pd.read_csv(input_path)
    
    # Create a new DataFrame with just LCLid and day columns
    result_df = df[['LCLid', 'day']].copy()
    
    # Get all the half-hourly columns
    hh_columns = [col for col in df.columns if col.startswith('hh_')]
    
    # Calculate the statistics for each row across all half-hourly columns
    result_df['energy_median'] = df[hh_columns].apply(np.median, axis=1)
    result_df['energy_mean'] = df[hh_columns].apply(np.mean, axis=1)
    result_df['energy_max'] = df[hh_columns].apply(np.max, axis=1)
    result_df['energy_count'] = df[hh_columns].count(axis=1)
    result_df['energy_std'] = df[hh_columns].apply(np.std, axis=1)
    result_df['energy_sum'] = df[hh_columns].apply(np.sum, axis=1)
    result_df['energy_min'] = df[hh_columns].apply(np.min, axis=1)
    
    # Save the result to a new CSV file
    result_df.to_csv(output_path, index=False)
    
    print(f"✔ Successfully processed {input_file}. Output saved to {output_file}.")
    return True

def merge_processed_files(block_range):
    """Merge all processed block files into a single CSV."""
    # Create an empty list to store all dataframes
    processed_files = []
    
    # Loop through the specified block range
    for i in block_range:
        processed_file = f'block_{i}_processed.csv'
        file_path = os.path.join(data_folder, processed_file)
        
        # Check if file exists
        if os.path.exists(file_path):
            processed_files.append(processed_file)
        else:
            print(f"✘ Warning: {processed_file} not found")
    
    if not processed_files:
        print("✘ No files found to merge")
        return
    
    # Read and concatenate all files at once
    all_dfs = pd.concat([pd.read_csv(os.path.join(data_folder, file)) for file in processed_files],
                        ignore_index=True)
    
    # Ensure columns are in the specified order
    columns = ['LCLid', 'day', 'energy_median', 'energy_mean', 'energy_max',
                'energy_count', 'energy_std', 'energy_sum', 'energy_min']
    
    all_dfs = all_dfs[columns]
    
    # Save the merged dataframe to a new CSV file
    output_path = os.path.join(data_folder, 'all_blocks_merged.csv')
    all_dfs.to_csv(output_path, index=False)
    
    print(f"✔ Merge complete! {len(processed_files)} processed blocks combined into 'all_blocks_merged.csv'")
    print(f"✔ Total rows in merged file: {len(all_dfs)}")

# Main execution
if __name__ == "__main__":
    # Define the range of blocks to process - ONLY EDIT THIS LINE
    block_range = range(0, 12)  # Process and merge blocks 0 to 10
    
    # Process each block
    processed_blocks = []
    for i in block_range:
        if process_block(i):
            processed_blocks.append(i)
    
    if processed_blocks:
        print(f"✔ Successfully processed blocks: {processed_blocks}")
        # Merge all processed files using the same range
        merge_processed_files(block_range)
    else:
        print("✘ No blocks were processed successfully.")
