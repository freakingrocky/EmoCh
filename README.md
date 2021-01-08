
# EmoCh- Emotion Analysis from Speech <img alt="Project stage: Development" src="https://img.shields.io/badge/Project%20Stage-Development-yellowgreen.svg" />

**There is a 1 Second Delay in classification**\
![Demo](https://i.imgur.com/MOA4gBr.gif)
## Emotion Analysis from Speech using Python

This software uses the tonal qualities of speech to determine the person's emotions. It is intended for use
in voice-only scenarios, since using Audio-Visual input can drastically improve accuracy and generalize the
AI's result in a wider range of inputs.

Python libraries such as `librosa`, `scikit`, `pyaudio`, `wave`, `numpy` are used in the production application.\
Additionally, python libraries such as `tensorflow` and `pydub` were used in the research stage.

### Installation

#### Method 1 (Not Recommended)
Windows users can download a zipped file from zippyshare ***(External)***\
&nbsp;&nbsp;&nbsp;&nbsp;[Click Here To Download Zip](https://www107.zippyshare.com/v/Y80pkhLE/file.html)\
**Though the major updates will be maintained on zippyshare link, smaller changes such as new models will not. Thus, I encourage you to use method 2.**
#### Method 2 (Recommended):
```cmd
git clone https://github.com/freakingrocky/EmoCh.git
cd EmoCh
python emoch.py
```
You can use the command-line-interface to test out any changes you made:
```cmd
python live_classifier.py
```


### Data Used for Training

[The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976)\
&nbsp;&nbsp;Only the audio is used from this dataset. This dataset's audio is in North American accent.\
[Toronto emotional speech set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)\
&nbsp;&nbsp;This dataset consists of 2 people of 26 and 67 years from the Toronto area (Canada)



### Research Approach

#### Feature Selection
Initially, `tonnetz` was also used but has been removed as it yielded low accuracy boost with high compute requirements.

The features are extracted and provided as input to the various models for training.

The extracted features are as follows:

- mfcc: Mel Frequency Cepstral Coefficient\
&nbsp;&nbsp;&nbsp;&nbsp;Significance: This can be used a heuristic metric for short term energy levels in voice.
- mel: Mel Frequency Scale\
&nbsp;&nbsp;&nbsp;&nbsp;Significance: The mel scale is a heuristic scale of pitches, as heard by people.
- chroma: Chroma\
&nbsp;&nbsp;&nbsp;&nbsp; Significance: This is a metric representing the octave of voice.

Once the features were extracted, various viable experiments were done to see which AI or ML model is best suited for the task.


#### Model Selection Experiments

Various Machine Learning and Artificial Intelligence models were trained and tested. Various different approaches were taken and data augmentation was done as per the model.

Here are there accuracy results on the randomly selected testing set:

- Support Vector Machine\
&nbsp;&nbsp;&nbsp;&nbsp;Best Accuracy Achieved: 77.26%
- Nearest Neighbor Classifier (One Node)\
&nbsp;&nbsp;&nbsp;&nbsp;Best Accuracy Achieved: 88.77%
- K-Nearest Neighbor Classifier (4 Nodes)\
&nbsp;&nbsp;&nbsp;&nbsp;Best Accuracy Achieved: 86.70%
- Naive Bayes Classifier\
&nbsp;&nbsp;&nbsp;&nbsp;Best Accuracy Achieved: 58.02%
- Deep Neural Network (with no user-defined features)\
&nbsp;&nbsp;&nbsp;&nbsp;Best Accuracy Achieved: 91.81%\
&nbsp;&nbsp;&nbsp;&nbsp;Extremely High Model Load Times; Requires high processing power. (Not Suitable for intended use-case)\
&nbsp;&nbsp;&nbsp;&nbsp;Also, this is probably the result of overfitting.
- Multi-Layer Perceptron (Convolutional Neural Network for internal classification)\
&nbsp;&nbsp;&nbsp;&nbsp;Best Accuracy Achieved: 89.62%

From the results above, it is clear that Multi-Layer Perceptron is best suited for the task at hand.

#### GUI
The GUI is made using `PySide2`, which is licensed under the [LGPL (GNU Lesser General Public License)](https://www.gnu.org/licenses/lgpl-3.0.en.html) license.

The splash screen checks the mic on the system and starts the main application, which immediately starts 1-sec delayed classification based on mic audio.

## Future Goals
- Train on More Data
- Train on data with noise **AND** noise removal AI integration
- Real Time audio stream classification (Current Problem is not enough samples in real-time)

## Limitations
- An AI/ML Classifier is only as good as the data it has trained on. The data used in this project is the open-source data designed for this use-case, which consists of people from western countries. The accuracy results are based entirely on this data.
- People from different geographical regions may have different tonal qualities in their voice for different emotions, this means there will be a bias based on geographical location and the data it was trained on.
- **1 Second Delay in classification**
