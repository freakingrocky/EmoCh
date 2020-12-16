"""Utility Functions for EmoCh."""
import pyaudio
p = pyaudio.PyAudio()

def get_audio_devices():
    """Return dictionary of audio devices on machine."""
    devices = [[], [], []]
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            info = p.get_device_info_by_host_api_device_index(0, i)
            devices[0].append(i)
            devices[1].append(info.get('name'))
            devices[2].append(info.get('defaultSampleRate'))
    return devices
