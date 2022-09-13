import os
import pywt
import pickle as pkl
import librosa
import numpy as np
from rich.progress import track

from tqdm import tqdm

from scipy import signal

from multiprocessing import Pool, Manager

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from scipy.io import wavfile
from skimage.transform import resize
from scipy.spatial.distance import cosine

# Goal is to clarify this file and make a large function that can generate the dataset with the preprocessing steps required (mel,log-scaled, etc.)

def load_raw_file(file):
	return librosa.load(file, sr=16000)[0]

def load_data_array_multi(paths, filename='dataset'):

	with Pool() as p:
		results = [p.apply_async(load_raw_file, args=(path,)) for path in paths]
		dataset = [p.get() for p in track(results, description='Computing spectrograms ...')]

	dataset = np.array(dataset)
	np.save(filename, dataset)

def correlation_matrix(projections):
	""" Projections must be of the form : (n_proj, x, y)"""
	correlation_matrix = np.corrcoef(np.array([np.matrix.flatten(p) for p in projections]))

	return correlation_matrix

def euclidian_distance(arr, brr):
	return np.linalg.norm(arr.flatten()-brr.flatten()) # Should reeturn Euclidian distance between matrices

def corrleation(arr, brr):
	return np.correlatee(arr.flatten(), brr.flatten)

def cosine_distance(arr, brr):
	return cosine(arr.flatten(), brr.flatten())

if __name__ == '__main__':

	import natsort as n

	pc = '/home/user/Documents/Antonin/Hearline/Clean_sounds_raw_60_human'
	pn = '/home/user/Documents/Antonin/Hearline/Noise_sounds_raw_60_human'

	files_noise = np.array([f for f in os.listdir(pn)])
	files_clean = np.array([f for f in os.listdir(pc)])

	files = np.intersect1d(files_noise, files_clean)
	paths_clean = [os.path.join(pc, f) for f in n.natsorted(files)]
	paths_noise = [os.path.join(pn, f) for f in n.natsorted(files)]

	load_data_array_multi(paths_clean, filename='raw_16k_clean.npy')
	load_data_array_multi(paths_noise, filename='raw_16k_noise.npy')



		
