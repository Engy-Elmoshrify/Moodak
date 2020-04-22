from PyQt5 import QtCore, QtGui, QtWidgets, uic,QtTest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QStyle
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import pickle
from time import sleep
from ANAD import predict
#loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

qtcreator_file  = "dialog.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)

class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def myFunc(self):
        self.Result_emo.setText(" ")
        QtTest.QTest.qWait(100)
        self.Recorder.setEnabled(False);
        fs = 16000  # Sample rate
        seconds = 3  # Duration of recording
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished        
        write('Sample.wav', fs, myrecording)  # Save as WAV file 
        Out_Emotion = predict.predict_record()
        print(Out_Emotion)
        self.Result_emo.setText("You are ...")
        QtTest.QTest.qWait(750)
        self.Result_emo.setText("You are " +Out_Emotion[0])
        self.Recorder.setEnabled(True);
    

        
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowFlags(
        QtCore.Qt.WindowStaysOnTopHint
        )
        self.Recorder.clicked.connect(self.myFunc)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())