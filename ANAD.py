import soundfile # to read audio file
import numpy as np
import scipy
import librosa # to extract speech features
import glob
import pandas as pd
from  scipy.sparse import csr_matrix
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
class predict :
        
        def extract_feature(file_name, **kwargs):
            """
            Extract feature from audio file `file_name`
                Features supported:
                    - MFCC (mfcc)
                    - Chroma (chroma)
                    - MEL Spectrogram Frequency (mel)
                    - Contrast (contrast)
                    - Tonnetz (tonnetz)
                e.g:
                `features = extract_feature(path, mel=True, mfcc=True)`
            """ 
            mfcc = kwargs.get("mfcc")
            chroma = kwargs.get("chroma")
            mel = kwargs.get("mel")
            contrast = kwargs.get("contrast")
            tonnetz = kwargs.get("tonnetz")
            with soundfile.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate
                #if chroma or contrast:
                #stft = np.abs(librosa.stft(X))
                audio=np.frombuffer(X,dtype=np.int16)
                stft = librosa.feature.melspectrogram(audio.astype('float32'), sr= sample_rate)
                result = np.array([])
                if mfcc:
                    #mfccs = np.mean(librosa.feature.mfcc(y=X, sr=16000, n_mfcc=40).T, axis=0)
                    mfc= np.mean(librosa.feature.mfcc(y=X, sr=16000, S=stft, n_mfcc=40).T,axis=0)
                    #result=result.reshape(40,334)
                    #result = result.reshape(1,total_length)
                    result = np.hstack((result, mfc))
                if chroma:
                    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=16000).T,axis=0)
                    result = np.hstack((result, chroma))
                if mel:  
                    mel = np.mean(librosa.feature.melspectrogram(y=X,S=stft, sr=16000).T,axis=0)
                    rows = len(mel)
                    columns = 1
                    total_length = rows * columns
                    #mel=mel.reshape(1,total_length)
                    result = np.hstack((result, mel))
                if contrast:
                    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=16000).T,axis=0)
                    result = np.hstack((result, contrast))
                if tonnetz:
                    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), S=stft,sr=sample_rate).T,axis=0)
                    tonnetz = np.mean(librosa.feature.tonnetz(y=X, sr=16000, chroma=chroma).T,axis=0)
                    result = np.hstack((result, tonnetz))
            return result
            
        def predict_record():
            X_record= []
            # extract speech features
            features = predict.extract_feature('Sample.wav', mfcc=True,chroma=True,mel=True,contrast=True,tonnetz=True)
            # add to data
            X_record.append(features)
            y_record = loaded_model.predict(np.array(X_record))
            return y_record

         