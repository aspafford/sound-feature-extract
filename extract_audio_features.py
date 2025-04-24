import argparse
import csv
from pathlib import Path # Using pathlib for modern path handling
import librosa
import numpy as np
import time # To time the process

# Define the number of features we extract (to ensure consistency)
NUM_MFCC = 13
NUM_CHROMA = 12

def main():
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description='Extract audio features from all .wav files in a directory and save to CSV.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing the WAV audio files.')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to the output CSV file.')
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)

    # Validate input directory
    if not input_path.is_dir():
        print(f"Error: Input directory not found or is not a directory: {input_path}")
        return

    # --- Prepare CSV Output ---
    # Define the header row for the CSV file
    # We flatten the MFCC and Chroma means into separate columns
    fieldnames = ['filepath', 'filename', 'duration_seconds', 'sample_rate', 'tempo'] + \
                 ['spectral_centroid_mean'] + \
                 [f'mfcc_mean_{i}' for i in range(NUM_MFCC)] + \
                 [f'chroma_mean_{i}' for i in range(NUM_CHROMA)]

    # Keep track of progress
    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Use 'with open' to ensure the file is properly closed
    # newline='' prevents extra blank rows in CSV on some systems
    # encoding='utf-8' is generally recommended
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() # Write the header row

        print(f"Scanning directory: {input_path}")
        # --- Iterate Through Files in Directory ---
        for file_path in input_path.iterdir():
            # Check if it's a file and has a .wav extension (case-insensitive)
            if file_path.is_file() and file_path.suffix.lower() == '.wav':
                print(f"Processing: {file_path.name}...")
                try:
                    # --- Load Audio ---
                    # Using resolve() to get the absolute path
                    abs_filepath = str(file_path.resolve())
                    y, sr = librosa.load(abs_filepath, sr=None, mono=True)
                    duration = librosa.get_duration(y=y, sr=sr)

                    # --- Feature Extraction (same logic as before) ---
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
                    mfccs_mean = np.mean(mfccs, axis=1)

                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spectral_centroid_mean = np.mean(spectral_centroid)

                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    # Ensure chroma has the correct number of dimensions before mean
                    if chroma.shape[0] != NUM_CHROMA:
                         # Handle cases where chroma might be computed differently (rare)
                         print(f"  Warning: Unexpected chroma shape {chroma.shape} for {file_path.name}. Skipping chroma mean.")
                         chroma_mean = [None] * NUM_CHROMA # Or np.nan
                    else:
                        chroma_mean = np.mean(chroma, axis=1)

                    # Tempo (using the working version for your librosa install)
                    # Add a check in case tempo detection fails or returns empty
                    tempo_array = librosa.beat.tempo(y=y, sr=sr)
                    primary_tempo = tempo_array[0] if (isinstance(tempo_array, np.ndarray) and tempo_array.size > 0) else None


                    # --- Prepare Data Row for CSV ---
                    row_data = {
                        'filepath': abs_filepath,
                        'filename': file_path.name,
                        'duration_seconds': round(duration, 3),
                        'sample_rate': sr,
                        'tempo': primary_tempo if primary_tempo is not None else '', # Handle None for CSV
                        'spectral_centroid_mean': round(spectral_centroid_mean, 3),
                    }

                    # Add flattened MFCC and Chroma means to the row dictionary
                    if len(mfccs_mean) == NUM_MFCC:
                        for i in range(NUM_MFCC):
                            row_data[f'mfcc_mean_{i}'] = round(mfccs_mean[i], 5)
                    else:
                         print(f"  Warning: Unexpected MFCC shape {mfccs.shape} for {file_path.name}. Skipping MFCCs.")
                         for i in range(NUM_MFCC): row_data[f'mfcc_mean_{i}'] = ''


                    if len(chroma_mean) == NUM_CHROMA:
                        for i in range(NUM_CHROMA):
                             # Check if chroma_mean[i] is None before rounding
                             row_data[f'chroma_mean_{i}'] = round(chroma_mean[i], 5) if chroma_mean[i] is not None else ''
                    else:
                        # This case handled by the shape check above, ensure columns exist
                        for i in range(NUM_CHROMA): row_data[f'chroma_mean_{i}'] = ''


                    # --- Write Row to CSV ---
                    writer.writerow(row_data)
                    processed_count += 1

                except Exception as e:
                    # Print error for the specific file and continue with the next
                    print(f"  ERROR processing {file_path.name}: {e}")
                    error_count += 1
            # else: # Optional: print if a file is skipped
            #     if file_path.is_file():
            #         print(f"Skipping non-WAV file: {file_path.name}")

    # --- Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} files.")
    print(f"Failed to process:    {error_count} files.")
    print(f"Total time taken:     {total_time:.2f} seconds.")
    print(f"Output saved to:      {output_path.resolve()}")

if __name__ == "__main__":
    # This ensures the main() function runs when the script is executed
    main()