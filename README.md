# Sound Feature Extraction

A tool for extracting audio features from multiple WAV files and saving them to a CSV file using the Librosa library.

## Installation

1. Clone this repository:
```
git clone https://github.com/aspafford/sound-feature-extract.git
cd sound-feature-extract
```

2. Install the required dependencies:
```
pip install librosa numpy
```

Note: Librosa relies on other libraries for loading audio files (like soundfile for WAV/FLAC and potentially audioread for MP3s). If you run into issues loading specific file types (especially MP3), you might need to install ffmpeg separately on your system.

## Usage

Run the script with command-line arguments:

```
python extract_audio_features.py --input-dir /path/to/your/audio/folder --output-file features_output.csv
```

Parameters:
- `--input-dir`: Directory containing your .wav files (required)
- `--output-file`: Path to your output CSV file (required)

The script will:
1. Scan the specified directory for WAV files
2. Process each WAV file to extract audio features
3. Save all features to a single CSV file
4. Display progress and a summary when complete

## Features Extracted

The script extracts and saves the following features for each WAV file:
- File path and filename
- Duration (in seconds)
- Sample rate
- Tempo (BPM)
- Spectral Centroid (mean)
- MFCCs (Mel Frequency Cepstral Coefficients) - 13 values
- Chroma Features - 12 values

## CSV Output Format

The CSV file contains one row per audio file with the following columns:
- `filepath`: Absolute path to the audio file
- `filename`: Just the filename
- `duration_seconds`: Length of the audio in seconds
- `sample_rate`: Sample rate in Hz
- `tempo`: Estimated tempo in BPM
- `spectral_centroid_mean`: Average spectral centroid
- `mfcc_mean_0` through `mfcc_mean_12`: Individual MFCC mean values
- `chroma_mean_0` through `chroma_mean_11`: Individual chroma mean values

## Example Output

```
--- Processing Complete ---
Successfully processed: 32 files.
Failed to process:    0 files.
Total time taken:     45.23 seconds.
Output saved to:      /Users/username/features_output.csv
```