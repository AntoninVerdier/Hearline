from mmap import mmap
import os
import json
import datetime
import argparse
import numpy as np
import natsort as n
import pickle as pkl
from sklearn import preprocessing as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D, Conv2DTranspose
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import Callback

import visualkeras
import matplotlib
from PIL import ImageFont
import matplotlib.pyplot as plt
from rich import print, traceback
traceback.install()
from rich.progress import track
import settings as s
import preproc as proc
from Models import Autoencoder, DenseMax
import tensorflow as tf

from tensorflow.keras import mixed_precision


mixed_precision.set_global_policy('mixed_float16')

# Define arguments for inline parsing
paths = s.paths()
params = s.params()

parser = argparse.ArgumentParser(description='Flags for model training')

parser.add_argument('--train', '-t', action='store_true',
                    help='Train the network')
parser.add_argument('--data_size', '-d', type=int, default=1,
                    help='Percentage of selected training data')
parser.add_argument('--predict', '-p', action='store_true',
                    help='')
parser.add_argument('--network', '-n', type=str,
                    help='Choose network type')
parser.add_argument('--batch_size', '-b', type=int, default=32,
                    help='Choose batch size')
parser.add_argument('--callbacks', '-c', action='store_true',
                    help='Choose if there is a tensorboard callback')
parser.add_argument('--max_n', '-mn', type=int, default=100,
                    help='Number of led to be lit up')
parser.add_argument('--visualize', '-v', action='store_true', 
                    help='flag to visualize a network')
parser.add_argument('--quicktest', '-qt', type=str, default=None,
                    help='Placeholder for name and description')
parser.add_argument('--epochs', '-e', type=int, default=150,
                    help='Number of epochs')
args = parser.parse_args()




   
# Execute training if inline argument is passed
if args.train:
    # Get time and date for record when saving
    today = datetime.date.today()
    time = datetime.datetime.now()

    # Create saving folder now so we can write callbacks in it
    today = today.strftime("%d%m%Y")
    time_str = time.strftime("%H%M%S")

    # Quick infos on the network for record
    if not args.quicktest:
      net_name = input('Name of the network > ')
      description = input('Small description of the current network training, for record > ')
    else:
      net_name, description = args.quicktest, args.quicktest

    logs = "new_logs/" + '{}_{}_{}'.format(today, time_str, net_name)
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                     histogram_freq =1,
                                                     profile_batch =(1, 10))
    
    input_dataset_file = 'noise_raw.npy'
    output_dataset_file = 'clean_raw.npy'


    # Distinguish between noisy input and clean reconstruction target
    # X_train = np.load(open(input_dataset_file, 'rb'), mmap_mode='r', allow_pickle=True).astype('float32')
    # X_train_c = np.load(open(output_dataset_file, 'rb'), mmap_mode='r', allow_pickle=True).astype('float32')

    # Datasets
    X_train = np.load(input_dataset_file, mmap_mode='r')
    X_train_c = np.load(output_dataset_file, mmap_mode='r')


    # Select the desired portion of the data and shuffle it
    shuffle_mask = np.random.choice(X_train.shape[0], int(args.data_size/100 * X_train.shape[0]), replace=False)
    X_train = X_train[shuffle_mask]
    X_train_c = X_train_c[shuffle_mask]

    # Add extra dimension to act as channem
    X_train = np.expand_dims(X_train, axis=2)
    X_train_c = np.expand_dims(X_train_c, axis=2)
    
    # This to enable fair splitting for convolution. Configured for spectrogram training
    if args.network: 

      input_shape = X_train.shape
      print(input_shape)


    # Create a validation set
    X_train, X_test, X_train_c, X_test_c = train_test_split(X_train, X_train_c, test_size=0.2, shuffle=True)

    X_train = tf.convert_to_tensor(X_train)
    X_train_c = tf.convert_to_tensor(X_train_c)



    # Create network class
    auto = Autoencoder('{net}'.format(net=args.network if args.network else 'dense'), input_shape, params.latent_size)

    # Retrive compiled model from network class
    autoencoder = auto.get_model()

    save_model_path = os.path.join(paths.path2Models, '{}_{}_{}'.format(today, time_str, net_name))
    if not os.path.exists(save_model_path):
      os.makedirs(save_model_path)
      os.makedirs(os.path.join(save_model_path, 'viz'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'data', 'sharp'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'data', 'blurred'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'img', 'sharp'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'img', 'blurred'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'spec', 'data'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'spec', 'img', 'both'))
      os.makedirs(os.path.join(save_model_path, 'predict', 'spec', 'img', 'indiv'))
      os.makedirs(os.path.join(save_model_path, 'Callbacks', 'Dat'))
      os.makedirs(os.path.join(save_model_path, 'Callbacks', 'Img'))
      os.makedirs(os.path.join(save_model_path, 'Performances', 'Img'))
      os.makedirs(os.path.join(save_model_path, 'Performances', 'Data'))

  

    history = autoencoder.fit(X_train, X_train_c,
                              use_multiprocessing=True,
                              epochs=args.epochs, 
                              batch_size=4,
                              callbacks=[tboard_callback])



    ################" SAVING MODEL INFO #########################

    # Add a timestamp to log files and model name so file is unique. 
    # Add ID to load it faster for further exp - WIP
    
    td = datetime.datetime.now() - time
    training_time = '{}h {}m {}s'.format(td.seconds//3600, (td.seconds//60)%60, td.seconds%60)

    autoencoder.save('Autoencoder_model_tcn')


    pkl.dump(history.history, open(os.path.join(save_model_path, 'model_history.pkl'), 'wb'))

    # Save arguments for record
    args_dict = vars(args)

    args_dict['name'] = net_name
    args_dict['desc'] = description
    args_dict['input_dataset_file'] = input_dataset_file
    args_dict['output_dataset_file'] = output_dataset_file
    args_dict['creation_date'] = today
    args_dict['creation time'] = time_str
    args_dict['training_time'] = training_time
    args_dict['epochs'] = params.epochs
    args_dict['best_loss'] = np.min(history.history['loss'])
    args_dict['end_loss'] = history.history['loss'][-1]

    # Add training time

    with open(os.path.join(save_model_path, 'metadata.json'), 'w') as f:
      json.dump(args_dict, f, indent=4)


    X_test = np.expand_dims(X_test, axis=2)
    y_pred = autoencoder.predict(X_test)
    np.save('prediction', [X_test, y_pred])


      



