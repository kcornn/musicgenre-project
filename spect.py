#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
import sklearn
import os.path
import sklearn
import os
import pickle


# In[2]:


tracks = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0,1])
small = tracks['set','subset'] == 'small'
small_tracks = tracks[small]


# In[3]:


small_tracks['track_id'] = small_tracks.index


# In[4]:


#small_tracks.head()


# In[5]:


train = small_tracks[small_tracks['set','split'] == 'training']
val = small_tracks[small_tracks['set','split'] == 'validation']
test = small_tracks[small_tracks['set','split'] == 'test']


# In[6]:


ROOT = "fma_small"


# In[7]:


#for root, subdirs, files in os.walk(ROOT):
#    print(root)


# In[8]:


# Make spectrogram based on track's id
def get_file(trackid):
    trackid = '{0:0=6d}'.format(trackid) # mp3 file 3 digits long
    folder = trackid[:3] # folder is first 3 digits
    filename = ROOT + '/' + folder + '/' + trackid + '.mp3'
    return filename

# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
def make_spect(trackid):
    filename = get_file(trackid)
    #print(filename)
    y, sr = librosa.load(filename) # (y, sampling rate of y)
    # https://stackoverflow.com/questions/46031397/using-librosa-to-plot-a-mel-spectrogram
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = spect.T
    return spect
    
def display_spect(spect):
    librosa.display.specshow(spect)


# In[9]:


spect = make_spect(5)


# In[12]:


def make_df(df):
    X = np.empty((0, 640, 128))
    y = []
    curr = 0
    for index, row in df.iterrows():
        curr += 1
        if curr == 1601:
            break
        #if curr < 1601:
        #   continue
        #if curr == 3201:
        #   break
        try:
            if curr % 100 == 0:
                print('number', curr)
            trackid = int(row['track_id'])
            curr_genre = str(row[('track', 'genre_top')])
            spect = make_spect(trackid)
            spect = spect[:640, :]
            X = np.append(X, [spect], axis=0)
            y.append(curr_genre)
        except:
            print('unable to process', curr)
            continue
    y = np.array(y)
    print(y)
    return X, y

print("Processing training set.")
X_train, y_train = make_df(train)

# In[ ]:


#X_train_pickle = open("X_train7.pickle","wb")
##pickle.dump(X_train, X_train_pickle)
#X_train_pickle.close()
#y_train_pickle = open("y_train7.pickle","wb")
#pickle.dump(y_train, y_train_pickle)
#y_train_pickle.close()

