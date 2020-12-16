# Audio Processing Imports
import librosa
import soundfile

# Other Imports
import numpy as np


def feature_extract(filename: str, single=True):
    """
    Extract features from audio required for model training.

    Parameters:
        - filename: path to the file
        - single: if True, returns one numpy object, else returns everything seperately. (Boolean)
    """
    with soundfile.SoundFile(filename) as sf:
        # Reading the file in and extracting required data
        raw_data = sf.read(dtype="float32")
        sr = sf.samplerate

        # Creating an empty numpy array to add data later on
        data = np.array([])

        # Calculating & Appending mfcc
        mfcc = np.mean(librosa.feature.mfcc(
            y=raw_data, sr=sr, n_mfcc=40).T, axis=0)
        data = np.hstack((data, mfcc))

        # Calculating & Appending chroma
        stft = np.abs(librosa.stft(raw_data))
        chroma = np.mean(librosa.feature.chroma_stft(
            S=stft, sr=sr,).T, axis=0)
        data = np.hstack((data, chroma))

        # Calculating & Appending mel
        mel = np.mean(librosa.feature.melspectrogram(
            raw_data, sr=sr).T, axis=0)
        data = np.hstack((data, mel))

        if single:
            return data

        return mfcc, chroma, mel
