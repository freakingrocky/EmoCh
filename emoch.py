"""Speech Emotion Analysis using Python."""

# GUI Imports
import sys
import platform
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime,
                            QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase,
                           QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

# Importing GUI Files
from splash_screen import Ui_SplashScreen
from main_window import Ui_MainWindow
from assets import *

# Imporitng Prediction Code
from live_classifier import get_emotion

# Importing Helper Functions
from utils import get_audio_devices
from live_classifier import start_stream, stop_stream

# Global Variables
PROGRESS = 0


class MainWindow(QMainWindow):
    """Main Window."""
    def __init__(self):
        """Setup & Control the window."""
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.pixmaps = {
            'angry': QPixmap(images['ANGRY']).scaled(650, 350, Qt.KeepAspectRatio, Qt.FastTransformation),
            'calm': QPixmap(images['CALM']).scaled(650, 350, Qt.KeepAspectRatio, Qt.FastTransformation),
            'apprehensive': QPixmap(images['APPREHENSIVE']).scaled(650, 350, Qt.KeepAspectRatio, Qt.FastTransformation),
            'elated': QPixmap(images['ELATED']).scaled(650, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
        }

        audioDevices = get_audio_devices()

        for item in audioDevices[1]:
            self.ui.InputSelect.addItem(item)

        self.ui.InputSelect.setCurrentIndex(1)

        self.ui.InputSelect.currentIndexChanged.connect(self.setaudiodevice)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.start)
        self.timer.start(1)

        self.show()

    def setaudiodevice(self):
        self.timer.stop()
        self.ui.label_2.setText("Just a sec...")
        stop_stream()
        start_stream(self.ui.InputSelect.currentIndex())
        self.timer.start(1)


    def start(self):
        emotion = get_emotion()
        self.ui.label_2.setText(emotion[0].upper())
        self.ui.Image.setPixmap(self.pixmaps[emotion[0]])


class SplashScreen(QMainWindow):
    """Splash Screen."""

    def __init__(self):
        """Setup the window."""
        # Initializing some required assets
        QMainWindow.__init__(self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

        try:
            start_stream()
        except:
            try:
                start_stream(0)
            except:

                QtCore.QTimer.singleShot(1500, lambda: self.ui.Name.setText("Error!"))
                sys.exit()

        # Setting Progress to 0 on startup
        self._set_progress(0)

        # Remove Title Bar
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Apply Drop Shadow
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 100))
        self.ui.Background.setGraphicsEffect(self.shadow)

        # Connecting progress bar
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._load)
        self.timer.start(20)

        # Outputing the GUI to the screen
        self.show()

    def _load(self):
        """Update Progress."""
        global PROGRESS

        # Updating Progress Bar
        self._set_progress(PROGRESS)
        # When Progress Bar Completes
        if PROGRESS > 100:
            self.timer.stop()

            # Show Main Window
            self.main = MainWindow()
            self.close()

        PROGRESS += 1

    def _set_progress(self, value):
        """Update progress bar in GUI."""
        # Converting values to float between 0 and 1
        progress = (100 - value) / 100.0

        # Creating GUI Update stylesheet
        new_progress_stylesheet = stylesheet.replace("{@02}", str(progress))

        # Updating GUI
        self.ui.Progress.setStyleSheet(new_progress_stylesheet)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SplashScreen()
    sys.exit(app.exec_())
