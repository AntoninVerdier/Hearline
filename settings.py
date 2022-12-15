import os
import tensorflow.keras

class paths():
	def __init__(self):
		self.path2Data = '/home/user/Documents/Antonin/Hearline/Data'
		self.path2Models = '/home/user/Documents/Antonin/Hearline/ModelSave'
		self.path2Output = '/home/user/Documents/Antonin/Hearline/OutputFig'
		self.path2OutputD = '/home/user/Documents/Antonin/Hearline/Latent'


class params():
	def __init__(self):
		self.gen_params = {'dim': (513,126),
							'batch_size': 256,
							'shuffle': False}
		self.specshape = self.gen_params['dim']
		self.latent_size = 100
		self.epochs = 100

		self.test_params = {'dim': (513,126),
							'batch_size': 1,
							'shuffle': False}
