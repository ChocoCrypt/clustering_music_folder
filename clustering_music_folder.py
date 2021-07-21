import os
import numpy as np
from sys import argv
from scipy.io import wavfile
import librosa
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns


if(len(argv) == 3):
    music_playlist = argv[1]
    clusters = int(argv[2])
    paths = [f"{music_playlist}/" +x for x in os.listdir(f"{music_playlist}")]
    tam = len(paths)

    cont = 1
    reduced_values = []
    #fill reduced values list with re scaled mel spectrograms of every song in the folder
    for i in paths:
        #reading files with librosa
        samples,sampling_rate = librosa.load(i)

        #creating spectrogram
        sgram = librosa.stft(samples)

        #generating mel spectrogram
        sgram_mag , _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag , sr = sampling_rate)

        #re scaling mel pectrograms with PCA
        pca = PCA(n_components = 2)
        pca.fit(mel_scale_sgram)
        values = pca.singular_values_
        reduced_values.append(values)
        print(f"{cont}/{tam} {i} has been added to reduced values ")
        cont += 1

    #changing reduced values list to numpy array
    reduced_values = np.array(reduced_values)
    #image_name
    fname = music_playlist.replace("/","")

#KMeans
    model = KMeans(n_clusters = clusters)
    model.fit(reduced_values)
    labels = model.predict(reduced_values)

#print values
    print("Kmeans" , "*"*50)
    for i in range(len(labels)):
        print(labels[i] , paths[i])

#Graphic model
    x = [x[0] for x in reduced_values]
    y = [x[1] for x in reduced_values]
    sns.scatterplot(x = x , y = y , hue = labels)
    plt.savefig(f"{fname}_kmeans.png")


#MeanShift
    model = MeanShift()
    model.fit(reduced_values)
    labels = model.labels_

#Graphic Model
    x = [x[0] for x in reduced_values]
    y = [x[1] for x in reduced_values]
    sns.scatterplot(x = x , y = y , hue = labels)
    plt.savefig(f"{fname}_meanshift.png")

#print values
    print("MeanShift" , "*"*50)
    for i in range(len(labels)):
        print(labels[i] , paths[i])


#Gaussian Mixture Models
    model  = GaussianMixture(n_components = clusters)
    model.fit(reduced_values)
    labels = model.predict(reduced_values)

#print values
    print("Gaussian Mixture Models:" , "*"*50)
    for i in range(len(labels)):
        print(labels[i] , paths[i])

#Graphic Model
    x = [x[0] for x in reduced_values]
    y = [x[1] for x in reduced_values]
    sns.scatterplot(x = x , y = y , hue = labels)
    plt.savefig(f"{fname}_gmm.png")

    print("Done")
else:
    print("must specify music playlist at argv1 and number of clusters with argv2")
