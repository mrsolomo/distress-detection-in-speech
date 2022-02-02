# By: SAYSON, Mary Raullette
#       1155156595@link.cuhk.edu.hk

import wave
import soundfile
import pickle
import matplotlib.pyplot as plt
import numpy as np
from struct import pack

from lib_audio import lib_audio_record, lib_audio_preproc
from lib_features import lib_features_extract

RATE = 16000
DATASET_PATH = "D:/School/CUHK/Research/Dataset/RAVDESS/"


def record_and_process_audio(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data_raw, data_processed = lib_audio_record()
    data_raw = pack('<' + ('h'*len(data_raw)), *data_raw)
    data_processed = pack('<' + ('h'*len(data_processed)), *data_processed)

    wf = wave.open(path + '_raw.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data_raw)
    wf.close()

    wf = wave.open(path + '_processed.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data_processed)
    wf.close()

def process_audio(path):
    with soundfile.SoundFile(path) as sound_file:
        data_raw = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        data_processed = lib_audio_preproc(data_raw)
    data_processed = pack('<' + ('h'*len(data_processed)), *data_processed)
    
    wf = wave.open(path[:-4] + '_processed.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(data_processed)
    wf.close()

# shows the sound waves
def visualize_audio(path_list):
    for count, path in enumerate(path_list):
        raw = wave.open(path)
        signal = raw.readframes(-1) # reads all the frames, -1 indicates all or max frames
        signal = np.frombuffer(signal, dtype ="int16")
        f_rate = raw.getframerate() # gets the frame rate
        # to Plot the x-axis in seconds
        # you need get the frame rate
        # and divide by size of your signal
        # to create a Time Vector
        # spaced linearly with the size
        # of the audio file
        time = np.linspace(
            0, # start
            len(signal) / f_rate,
            num = len(signal)
        )
        plt.figure(count + 1) # creates a new figure
        plt.title("Sound Wave (" + path + ")")
        plt.xlabel("Time")
        plt.plot(time, signal)
    
    plt.show()  # plt.savefig('filename')
 

if __name__ == "__main__":
    # path = "audio/hello.wav"
    # process_audio(path)
    # visualize_audio([path, path[:-4] + "_processed.wav"])
    
    print("Recording...")
    
    filename = "./audio/test"
    record_and_process_audio(filename)
    #visualize_audio([filename + '_raw.wav', filename + '_processed.wav'])
    features = lib_features_extract(filename + '_processed.wav', \
            mfcc=True, chroma=False, mel=False, contrast=False, tonnetz=False,\
            rms=False, zcr=False ).reshape(1, -1)
    
    # filename = DATASET_PATH + "Actor_01/03-01-05-02-01-02-01.wav"
    # features = lib_features_extract(filename, \
    #         mfcc=True, chroma=False, mel=False, contrast=False, tonnetz=False,\
    #         rms=False, zcr=False ).reshape(1, -1)
    
    model = pickle.load(open("model/linear_svc.model", "rb"))
    
    # predict
    result = model.predict(features)

    # result
    print("Result", result)