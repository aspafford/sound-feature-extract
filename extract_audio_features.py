import librosa
import numpy as np
import os
import argparse

def extract_features(audio_file_path):
    """Extract audio features from the given audio file."""
    try:
        print(f"Loading audio file: {audio_file_path}...")
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)  # sr=None keeps original sample rate
        print(f"Audio loaded successfully. Sample rate: {sr} Hz, Duration: {len(y)/sr:.2f} seconds")

        # --- Feature Extraction ---

        # 1. MFCCs (Mel Frequency Cepstral Coefficients) - Captures timbre/texture
        print("Calculating MFCCs...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        print(f"  Shape of MFCC matrix: {mfccs.shape}")
        print(f"  Mean MFCCs across time: {mfccs_mean}")

        # 2. Spectral Centroid - Indicates "brightness"
        print("Calculating Spectral Centroid...")
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        print(f"  Shape of Spectral Centroid: {spectral_centroid.shape}")
        print(f"  Mean Spectral Centroid: {spectral_centroid_mean:.2f}")

        # 3. Chroma Features - Relates to pitch classes (C, C#, D, etc.) - Good for harmony
        print("Calculating Chroma Features...")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        print(f"  Shape of Chroma matrix: {chroma.shape}")
        print(f"  Mean Chroma features: {chroma_mean}")

        # 4. Tempo (BPM - Beats Per Minute)
        print("Estimating Tempo...")
        # Use the older path valid for your Librosa version
        # AND fix the unpacking error by only assigning to 'tempo'
        tempo = librosa.beat.tempo(y=y, sr=sr)

        # Access the tempo value(s) from the returned NumPy array
        primary_tempo = tempo[0] if len(tempo) > 0 else None

        print(f"  Estimated Tempo(s): {tempo}")
        print(f"  Primary Estimated Tempo: {primary_tempo} BPM")

        # Return the extracted features as a dictionary
        features = {
            'mfccs_mean': mfccs_mean,
            'spectral_centroid_mean': spectral_centroid_mean,
            'chroma_mean': chroma_mean,
            'tempo': tempo[0] if isinstance(tempo, np.ndarray) else tempo
        }

        print("\n--- Feature Extraction Complete ---")
        return features

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract audio features from sound files.')
    parser.add_argument('audio_file', help='Path to the audio file to analyze')
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' does not exist.")
        return
    
    extract_features(args.audio_file)

if __name__ == "__main__":
    main()