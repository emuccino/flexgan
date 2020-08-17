import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD

from dataclasses import dataclass

from .utils import *
from .model import GeneratorModel

def generator(*args, **kwargs):

	"""
		Function for initalizing DataGenerator data generator.

		Args:
			dataframe (pd.DataFrame): A pandas data frame containing data that will be synthetically modeled.
				Alternatively, csv_path can be provided.
			csv_path (str): Path to csv data to be used to train the synthetic data generator. Alternatively, dataframe
				can be provided.
			target_names (list): The number of training epochs without improvement to allow before halting training.
				Larger patience generally produces better results but increases model training time.
			target_names (list): List of target feature names. Specifying target names produces more pronounced
				decision boundaries within the synthetic data and is helpful when synthetic data is planning to be
				used for classification or regression tasks.
			numerical_names (list): List of numerical feature names. Specifying numerical names can help ensure that
				data is processed appropriately, however it is not required.
			categorical_names (list): List of categorical feature names. Specifying numerical names can help ensure that
				data is processed appropriately, however it is not required.
			n_latent (int): Size of the latent vector space used to sample data. Larger n_latent values generally produce
				better results but can use more memory and slow down training time. The default is 16. 
			n_neurons (int): Number of neurons used within layers of the data generator neural networks. Larger n_neurons
				values generally produce better results but can use more memory and slow down training time. The default is 64.
			n_layers (int): Number of layers used within layers of the data generator neural networks. Larger n_layers
				values generally produce better results but can use more memory and slow down training time. The default is 64.
			model_path (str): Path to load pretrained data generator model.

		Returns:
			DataGenerator: Initalized class object for synthetic data generation. Call .train_data_generator to begin training the
				synthetic data generator model. Call .generate_data to generate synthetic data. Call .save_data_generator to
				save a trained synthetic data generator model for future use.
		"""

	return DataGenerator(*args, **kwargs)


class DataGenerator():

	def __init__(self, dataframe=None, target_names=None, numerical_names=None, categorical_names=None,
		csv_path=None, n_latent=16, n_neurons=128, n_layers=4, model_path=None):

		if csv_path:
			dataframe = pd.read_csv(csv_path)

		data = dict(dataframe)

		self._original_dataframe = dataframe.copy()

		self._columns = dataframe.columns
		self._n_latent = n_latent

		self._dtypes = self._sort_dtypes(data, numerical_names, categorical_names, target_names)
		self._processors = self._get_processors(data)
		self._data = self._preprocess_data(data)
		self._n_samples = len(dataframe)
		self._index = np.arange(self._n_samples)

		self._model = GeneratorModel(self._processors, self._dtypes, n_latent, n_neurons, n_layers, model_path)

		return


	def train(self, batch_size=1024, patience=2000):

		"""
		Function for training synthetic data generator model.

		Args:
			batch_size (int): The number of data samples to use in each training batch.
				Larger batch_size generally produce better results but use more memory. The default is 2000.
			patience (int): The number of training epochs without improvement to allow before halting training.
				Larger patience generally produces better results but increases model training time.

		Returns:
			None
		"""

		if self._dtypes['targets']:
			self._train_classifier(batch_size, patience)

		n_batches = int(np.ceil(self._n_samples / (batch_size * 4)))

		for epoch in range(patience):
			generator_loss = self._train_epoch(batch_size, n_batches)
		
		current_weights = self._model.gan.get_weights()
		self._weights_sum = current_weights
		self._weights_sum_squared = [w**2 for w in current_weights]
		
		optimizer = SGD(0.01)
		self._model.discriminator.trainable = True
		self._model.discriminator.compile(loss=self._model.discriminator.loss, optimizer=optimizer)
		self._model.discriminator.trainable = False
		self._model.gan.compile(loss=self._model.gan.loss, optimizer=optimizer)

		learning_rate_schedule = .1**np.linspace(2,3,patience)

		epoch = 0

		best_std = np.inf
		while True:
			learning_rate = learning_rate_schedule[epoch%patience]
			K.set_value(self._model.gan.optimizer.learning_rate, learning_rate)

			self._train_epoch(batch_size, n_batches)

			current_weights = self._model.gan.get_weights()
			weights_sum = [aw + cw for aw,cw in zip(weights_sum, current_weights)]
			weights_sum_squared = [aws + cw**2 for aws,cw in zip(weights_sum_squared, current_weights)]

			epoch += 1

			if epoch % patience == 0:
				average_weights = np.hstack([w.flatten() for w in weights_sum])/patience
				average_squared_weights = np.hstack([w.flatten() for w in weights_sum_squared])/patience
				average_std = np.mean(np.sqrt(np.max(average_squared_weights - average_weights**2, 0)))

				self._model.gan.set_weights([w/patience for w in weights_sum])

				if average_std < best_std:
					best_std = average_std
				else:
					return

				current_weights = self._model.gan.get_weights()
				weights_sum = current_weights
				weights_sum_squared = [w**2 for w in current_weights]


	def generate_data(self, n_samples=None, to_csv=None):

		"""
		Function for generating synthetic data.

		Args:
			n_samples (int): The number of data samples to generate. The default is to produce the same amount as the original data.
			to_csv (str): The path to save a csv file of the generated data.

		Returns:
			pandas.Dataframe: A dataframe cointaining synthetic data.
		"""

		if not n_samples:
			n_samples = self._n_samples

		synthetic_data = self._generate_synthetic_samples(n_samples)
		synthetic_data = self._postprocess_data(synthetic_data)
		synthetic_dataframe = pd.DataFrame(pd.DataFrame({key:value.flatten() for key,value in synthetic_data.items()}))[self._columns]

		if to_csv:
			synthetic_dataframe.to_csv(to_csv)

		return synthetic_dataframe


	def save_model(self,path):
		
		"""
		Function for saving a trained synthetic data generator model for future use.

		Args:
			path (str): The path to save the synthetic data generator model. Model is saved
				as an h5 file.

		Returns:
			None
		"""

		self._model.gan.save_weights(path)

		return


	def _verify_columns(self, data, dtypes):

		for name in dtypes['targets']:
			if not name in data:
				raise ColumnNotFound(f'{name} is given as a target column name but is not found in the data.')

		for name in dtypes['numerical']:
			if not name in data:
				raise ColumnNotFound(f'{name} is given as a numerical column name but is not found in the data.')

		for name in dtypes['categorical']:
			if not name in data:
				raise ColumnNotFound(f'{name} is given as a categorical column name but is not found in the data.')

		return


	def _sort_dtypes(self, data, numerical_names, categorical_names, target_names):

		dtypes = {'numerical': none_to_set(numerical_names),
				  'categorical': none_to_set(categorical_names),
				  'features': set(),
				  'targets': none_to_set(target_names)}

		self._verify_columns(data, dtypes)

		for name, values in data.items():
			if name not in dtypes['targets']:
				dtypes['features'].update({name})

			if name not in dtypes['numerical'] | dtypes['categorical']:
				if np.issubdtype(values.dtype, np.number):
					dtypes['numerical'].update({name})

				else:
					dtypes['categorical'].update({name})

		return dtypes


	def _get_processors(self, data):

		processors = {}

		for name, values in data.items():
			if name in self._dtypes['numerical']:
				processor = NumericalProcesssor(name, values)
				processors[name] = processor

			if name in self._dtypes['categorical']:
				processor = CategoricalProcesssor(name, values)
				processors[name] = processor

		return processors


	def _get_nan_gates(self, data):

		for name in self._dtypes['numerical'] | (self._dtypes['categorical'] & self._dtypes['targets']):
			gate_name = f'gate.{name}'
			data[gate_name] = pd.Series(data[name]).notna().astype(int).values.reshape(-1,1)

		return data


	def _preprocess_data(self, data):

		processed_data = self._get_nan_gates(data)
		processed_data.update({name: processor.preprocess(processed_data[name]) for name, processor in self._processors.items()})

		return processed_data


	def _postprocess_data(self, data, keep_index=False):

		for name in self._dtypes['numerical']:
			gate_name = f'gate.{name}'
			data[name] = data[name] * data[gate_name]

		processed_data = {name: processor.postprocess(data[name], keep_index=keep_index) for name, processor in self._processors.items()}

		return processed_data


	def _get_classifier_data(self):

		np.random.shuffle(self._index)
		n_train = int(self._n_samples * .8)

		x_train = {}
		y_train = {}

		x_test = {}
		y_test = {}

		for indx, (x, y) in zip([slice(0,n_train), slice(n_train,self._n_samples)], [(x_train, y_train), (x_test, y_test)]):

			for name in self._dtypes['numerical']:
				values = self._data[name][self._index[indx]]
				x[name] = values

				if name in self._dtypes['targets']:
					output_name = f'classifier.{name}'
					y[output_name] = values
					
					gate_name = f'gate.{name}'
					x[gate_name] = self._data[gate_name][self._index[indx]]

			for name in self._dtypes['categorical']:
				values = to_categorical(self._data[name][self._index[indx]], num_classes=self._processors[name].n_tokens+1)[:,1:]
				x[name] = values

				if name in self._dtypes['targets']:
					output_name = f'classifier.{name}'
					y[output_name] = values

					gate_name = f'gate.{name}'
					x[gate_name] = self._data[gate_name][self._index[indx]]

		for name in self._dtypes['categorical'] & self._dtypes['targets']:
			output_name = f'classifier.{name}'
			y_train[output_name] = self._smooth(y_train[output_name], name)

		return x_train, y_train, x_test, y_test


	def _smooth(self, values, name):

		n_tokens = self._processors[name].n_tokens
		smoothing_limit = n_tokens / (10 * (n_tokens - 1))
		random_smoothing = np.random.uniform(high=smoothing_limit, low=0.,size=(len(values),1))
		smoothed_values = values * (1.0 - random_smoothing) + (random_smoothing / n_tokens)

		return smoothed_values


	def _train_classifier(self, batch_size, patience):

		x_train, y_train, x_test, y_test = self._get_classifier_data()
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
		self._model.classifier.fit(x=x_train, y=y_train, batch_size=batch_size, validation_data=(x_test, y_test), epochs=1000000, verbose=0, callbacks=[early_stop])
		self._model.classifier.evaluate(x=x_test, y=y_test)

		return


	def _get_generator_batch(self, n, distribution='standard'):

		self._choice = np.random.choice(self._index,size=n, replace=True)

		x = {}
		y = {}

		z_shape = (int(np.ceil(n/2)), self._n_latent,)

		if distribution == 'standard':
			z = tf.random.normal(z_shape, mean=0.0, stddev=1.0).numpy()

		elif distribution == 'truncated':
			z = tf.random.truncated_normal(z_shape, mean=0.0, stddev=1.0).numpy()
		
		z = np.vstack([z,-z])
		np.random.shuffle(z)

		x['z'] = z[:n]

		for name in self._dtypes['numerical']:
			x[name] = self._data[name][self._choice]

			gate_name = f'gate.{name}'
			x[gate_name] = self._data[gate_name][self._choice]

			if name in self._dtypes['targets']:
				classifier_name = f'gan.classifier.{name}'
				y[classifier_name] = x[name]

			for name in self._dtypes['categorical']:
				x[name] = to_categorical(self._data[name][self._choice], num_classes=self._processors[name].n_tokens+1)[:,1:]

				if name in self._dtypes['targets']:
					classifier_name = f'gan.classifier.{name}'
					y[classifier_name] = x[name]

					gate_name = f'gate.{name}'
					x[gate_name] = self._data[gate_name][self._choice]       

		discrimination = np.ones(shape=(n,1))
		y['gan.discriminator.discrimination.real'] = discrimination
		y['gan.discriminator.discrimination.synthetic'] = 1.5-discrimination
		y['gan.discriminator.discrimination'] = y['gan.discriminator.discrimination.real']
		return x, y
		
	
	def _get_synthetic_samples(self, n, distribution='standard'):
		
		latent_x, _  = self._get_generator_batch(n, distribution=distribution)
		generator_input = {name:latent_x[name] for name in self._model.generator.input_names}
		x = self._model.generator(generator_input)
		x = {key:value.numpy() for key, value in x.items()}
		y = {}

		for name in self._dtypes['numerical'] & self._dtypes['targets']:
			x[name] = latent_x[name]

		for name in self._dtypes['categorical']:
			x[name] = latent_x[name]

		discrimination = np.zeros(shape=(n,1))
		y['discriminator.discrimination.real'] = discrimination+0.5
		y['discriminator.discrimination.synthetic'] = 1-discrimination
		y['discriminator.discrimination'] = y['discriminator.discrimination.real']

		return x, y


	def _generate_synthetic_samples(self, n):

		x, _ = self._get_synthetic_samples(n, distribution='truncated')

		for name in self._dtypes['numerical']:
			gate_name = f'gate.{name}'
			x[gate_name] = self._data[gate_name][self._choice]

		return x


	def _generate_real_samples(self, n):

		x, _ = self._get_real_samples(n)

		for name in self._dtypes['numerical']:
			gate_name = f'gate.{name}'
			x[gate_name] = self._data[gate_name][self._choice]

		return x


	def _get_real_samples(self, n):
		
		self._choice = np.random.choice(self._index,size=n, replace=True)

		x = {}
		y = {}

		for name in self._dtypes['numerical']:
			x[name] = self._data[name][self._choice]

		for name in self._dtypes['categorical']:
			x[name] = to_categorical(self._data[name][self._choice], num_classes=self._processors[name].n_tokens+1)[:,1:]

		discrimination = np.ones(shape=(n,1))
		y['discriminator.discrimination.real'] = discrimination
		y['discriminator.discrimination.synthetic'] = 1.5 - discrimination
		y['discriminator.discrimination'] = y['discriminator.discrimination.real']

		return x, y


	def _get_discriminator_batch(self, batch_size, distribution='standard'):

		x = {}
		y = {}

		x_synthetic, y_synthetic = self._get_synthetic_samples(batch_size//2, distribution=distribution)
		x_real, y_real = self._get_real_samples(batch_size//2)

		for key in x_real.keys():
			x[key] = np.vstack([x_synthetic[key], x_real[key]])
		for key in y_real.keys():
			y[key] = np.vstack([y_synthetic[key], y_real[key]])

		return x, y


	def _train_batch(self, batch_size):

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		x_generator, y_generator = self._get_generator_batch(batch_size)
		self._model.gan.train_on_batch(x_generator, y_generator)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)

		x_generator, y_generator = self._get_generator_batch(batch_size)
		self._model.gan.train_on_batch(x_generator, y_generator)
		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		x_generator, y_generator = self._get_generator_batch(batch_size)
		self._model.gan.train_on_batch(x_generator, y_generator)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		x_generator, y_generator = self._get_generator_batch(batch_size)
		self._model.gan.train_on_batch(x_generator, y_generator)

		return


	def _train_epoch(self, batch_size, n_batches):

		for _ in range(n_batches):
			self._train_batch(batch_size)

		return