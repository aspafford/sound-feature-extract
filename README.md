# Sound Feature Extraction

A tool for extracting audio features from sound files using the Librosa library.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/sound-feature-extract.git
cd sound-feature-extract
```

2. Install the required dependencies:
```
pip install librosa numpy
```

Note: Librosa relies on other libraries for loading audio files (like soundfile for WAV/FLAC and potentially audioread for MP3s). If you run into issues loading specific file types (especially MP3), you might need to install ffmpeg separately on your system.

## Usage

Run the script with the path to your audio file:

```
python extract_audio_features.py /path/to/your/audio/file.wav
```

The script will extract and display the following features:
- MFCCs (Mel Frequency Cepstral Coefficients)
- Spectral Centroid
- Chroma Features
- Tempo (BPM)

## Example Output

```
Loading audio file: my_vocal.wav...
Audio loaded successfully. Sample rate: 44100 Hz, Duration: 3.50 seconds
Calculating MFCCs...
  Shape of MFCC matrix: (13, 304)
  Mean MFCCs across time: [ 0.23 -1.45  0.67  0.12 -0.34  0.56  0.01  0.11 -0.12  0.05  0.02 -0.01  0.01]
Calculating Spectral Centroid...
  Shape of Spectral Centroid: (1, 304)
  Mean Spectral Centroid: 2345.67
Calculating Chroma Features...
  Shape of Chroma matrix: (12, 304)
  Mean Chroma features: [0.12 0.09 0.15 0.07 0.23 0.08 0.05 0.03 0.07 0.04 0.11 0.06]
Estimating Tempo...
  Estimated Tempo: [120.] BPM

--- Feature Extraction Complete ---
```