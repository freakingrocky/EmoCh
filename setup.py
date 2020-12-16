import cx_Freeze
import sys
import os
import PySide2

plugins_path = os.path.join(PySide2.__path__[0], "plugins")

base = None

if sys.platform == 'win32':
	base = "Win32GUI"

executables = [
	cx_Freeze.Executable("emoch.py", base=base)
]

cx_Freeze.setup(
	name = "EmoCh- Speech Emotion Analysis",
	options = {"build_exe": {
		"zip_include_packages": ["PySide2", "platform", "librosa", "soundfile", 'numpy', 'joblib', 'pyaudio', 'wave', 'termcolor'],
		"include_files": [os.path.join(plugins_path, "platforms")]}},
	version = "1.0.0",
	description = "Emotion Analysis from Speech using Python",
	executables = executables
)
