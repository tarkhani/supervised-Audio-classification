import librosa
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


def SaveAsAudioClassificationModel(DataSetPath, NumMfcc=25, NumFft=2048, HopLength=900, NumSeg=20, SampleRate=22050,
                                   DurationOfTrack=30):
    data = {
        "genre": [],
        "MFCC": [],
        "LABEL": []
    }

    SamplePerSegmnet = int((SampleRate * DurationOfTrack) / NumSeg)
    expected_legnth_mfcc = math.ceil(SamplePerSegmnet / HopLength)

    genres = []
    for i, (DirPath, DirNames, FileNames) in enumerate(os.walk(DataSetPath)):
        if DirPath != DataSetPath:

            genre_label = DirPath.split("\\")[-1]
            if genre_label not in genres:
                genres.append(genre_label)
            label_index = genres.index(genre_label)

            for fileName in FileNames:
                FilePath = os.path.join(DirPath, fileName)
                Signal, SampleRate = librosa.load(FilePath, sr=SampleRate)

                for segment in range(NumSeg):
                    startOFSignal = SamplePerSegmnet * segment
                    EndOfSignal = startOFSignal + SamplePerSegmnet
                    mfcc = librosa.feature.mfcc(Signal[startOFSignal:EndOfSignal], SampleRate,
                                                n_mfcc=NumMfcc, hop_length=HopLength, n_fft=NumFft)
                    mfcc = mfcc.T
                    if len(mfcc) == expected_legnth_mfcc:
                        data["MFCC"].append(mfcc.tolist())
                        data["LABEL"].append(label_index)

    input_data = np.array(data["MFCC"])
    output_data = np.array(data["LABEL"])

    input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.3)
    input_train, input_validation, output_train, output_validation = train_test_split(input_train, output_train, test_size=0.2)

    input_train = input_train[..., np.newaxis]
    input_test = input_test[..., np.newaxis]
    input_validation = input_validation[..., np.newaxis]

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(expected_legnth_mfcc, NumMfcc, 1)))
    model.add(keras.layers.MaxPool2D((5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(len(genres), activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(input_train, output_train, validation_data=(input_validation, output_validation), batch_size=64, epochs=100)
    test_error, test_accuracy = model.evaluate(input_test, output_test, verbose=1)
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    model.save("model")
