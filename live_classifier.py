"""Live Classifications from mic."""
# Importing other libraries
import joblib
import numpy as np
import sys
from termcolor import cprint
import os

# Audio Imports
import pyaudio
import wave
import librosa

# Opening Model
MLP = joblib.load("./assets/MLP.joblib")

# Audio Capture Parameters
CHUNKSIZE = 1024
RATE = 44100

p = pyaudio.PyAudio()


def start_stream(index=1):
    """Initializing PyAudio Capture Stream"""
    global stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNKSIZE,
                    input_device_index=index)


def stop_stream():
    """Terminate stream."""
    stream.stop_stream()
    stream.close()


def get_features(filename: str):
    """
    Extract features from audio required for model training.

    return in required format.
    """
    # Reading the file in and extracting required data
    raw_data, sr = librosa.load(filename)

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

    x = []
    x.append(data)
    return np.array(x)


color_table = {
    'calm': 'yellow',
    'angry': 'red',
    'apprehensive': 'blue',
    'elated': 'green'
}


def get_emotion():
    """Predict emotion."""
    global stream
    frames = []
    for _ in range(0, int(RATE / CHUNKSIZE * 1)):
        data = stream.read(CHUNKSIZE)
        frames.append(data)
    with wave.open("emoch.tmp", 'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        file.setframerate(RATE)
        file.writeframes(b''.join(frames))
    features = get_features("emoch.tmp")
    return MLP.predict(features)


if __name__ == "__main__":
    try:
        start_stream()
        cprint("[+] Ready!!!", 'green', attrs=['bold'])
        cprint("Warning: There is a one second delay", 'red', attrs=['bold'])
        print()
        while True:
            emotion = get_emotion()
            sys.stdout.write("\033[K")
            cprint(emotion[0], color_table[emotion[0]], end='\r')
    except KeyboardInterrupt:
        cprint("[-] Closing...", 'red')
        if os.path.exists("emoch.tmp"):
            os.remove("emoch.tmp")
        stop_stream()
        p.terminate()
