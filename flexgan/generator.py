import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from .utils import *
from .model import FlexGANModel



def generator(*args, **kwargs):

	"""
		Function for initalizing DataGenerator.

		Args:
			dataframe (pd.DataFrame): A pandas data frame containing data that will be synthetically modeled.
				Alternatively, csv_path can be provided.
			target_names (list): List of target feature names. Specifying target names produces more pronounced
				decision boundaries within the synthetic data and is helpful when synthetic data is planning to be
				used for classification or regression tasks.
			numerical_names (list): List of numerical feature names. Specifying numerical names can help ensure that
				data is processed appropriately, however it is not required.
			categorical_names (list): List of categorical feature names. Specifying numerical names can help ensure that
				data is processed appropriately, however it is not required.
			n_latent (int): Size of the latent vector space used to sample data. Larger n_latent values generally produce
				better results but can use more memory and slow down training time. The default is 32. 
			n_neurons (int): Number of neurons used within layers of the data generator neural networks. Larger n_neurons
				values generally produce better results but can use more memory and slow down training time. The default is 128.
			n_layers (int): Number of layers used within layers of the data generator neural networks. Larger n_layers
				values generally produce better results but can use more memory and slow down training time. The default is 4.
			model_path (str): Path to load pretrained data generator model.

		Returns:
			DataGenerator: Initalized class object for synthetic data generation.
				Call .train() to begin training the synthetic data generator model.
				Call .generate_data() to generate synthetic data.
				Call .save_model() to save a trained synthetic data generator model for future use.
		"""

	return DataGenerator(*args, **kwargs)


class DataGenerator():

	def __init__(self, dataframe=None, target_names=None, numerical_names=None, categorical_names=None,
		csv_path=None, n_latent=32, n_neurons=128, n_layers=4, model_path=None):

		if csv_path:
			dataframe = pd.read_csv(csv_path)
		else:
			dataframe = pd.DataFrame(dataframe, copy=True)

		self._columns = dataframe.columns
		self._n_samples = len(dataframe)
		self._index = np.arange(self._n_samples)

		self._dtypes = self._get_dtypes(dataframe, numerical_names, categorical_names, target_names)
		self._processors = self._get_processors(dataframe)
		self._dataframe = self._setup_dataframe(dataframe)

		self._model = FlexGANModel(self._processors, self._dtypes, n_latent, n_neurons, n_layers, model_path)

		return


	def train(self, batch_size=1024, patience=500):

		"""
		Function for training synthetic data generator model.

		Args:
			batch_size (int): The number of data samples to use in each training batch.
				Larger batch_size generally produce better results but use more memory. The default is 1024.
			patience (int): The number of training epochs without improvement to allow before halting training.
				Larger patience generally produces better results but increases model training time. The default is 500.

		Returns:
			None
		"""

		batch_size = int(batch_size)
		patience = int(patience)
		n_batches = int(np.ceil(self._n_samples / (batch_size * 4)))
		best_loss = np.inf
		epoch = 0

		learning_rate_schedule = .1**np.linspace(2,3,patience)

		if self._dtypes['targets']:
			self._train_classifier(batch_size, patience)

		while True:
			learning_rate = learning_rate_schedule[epoch%patience]
			K.set_value(self._model.gan.optimizer.learning_rate, learning_rate)
			K.set_value(self._model.discriminator.optimizer.learning_rate, learning_rate)

			loss = self._train_epoch(batch_size, n_batches)

			epoch += 1
			if epoch % patience == 0:
				if loss < best_loss:
					best_loss = loss
					weights = self._model.gan.get_weights()
				else:
					self._model.gan.set_weights(weights)
					return


	def generate_data(self, n_samples=None, class_labels=None, to_csv=None):

		"""
		Function for generating synthetic data.

		Args:
			n_samples (int): The number of data samples to generate. If None, generated sample count will be the
				equal to the original data sample count, or if. n_samples will be ignored if class_labels are supplied.
				samples as is in the original data. 
			class_labels (dict): A dictionary-like object (e.g. pandas.DataFrame) that contains categorical and/or target labels
				for custom class label distributions within generated data. If None, categorical and target values are randomly
				drawn from the original class label distribution.
			to_csv (str): Path and filename to save synthetic data as csv. (optional)

		Returns:
			pandas.DataFrame: A dataframe cointaining synthetic data.
		"""

		if class_labels is not None:
			self._verify_class_labels(class_labels)
			class_labels = self._preprocess(pd.DataFrame(class_labels, copy=True))
			n_samples = len(class_labels)

		elif not n_samples:
			n_samples = self._n_samples

		data, _ = self._get_synthetic_samples(n_samples, class_labels=class_labels)

		df = self._to_dataframe(data)
		df = self._apply_gates(df)
		df = self._postprocess(df)
		df = df[self._columns]

		if to_csv:
			df.to_csv(to_csv, index=False)

		return df


	def save_model(self,path):
		
		"""
		Function for saving a trained synthetic data generator model for future use.

		Args:
			path (str): Path and file name to save the synthetic data generator model. E.g. 'path/my_flexgan_model.h5'

		Returns:
			None
		"""

		self._model.gan.save_weights(path)

		return


	def _get_classifier_data(self):

		np.random.shuffle(self._index)
		n_train = int(self._n_samples * .8)

		x_train = {}
		y_train = {}

		x_test = {}
		y_test = {}

		for indx, (x, y) in zip([slice(0,n_train), slice(n_train,self._n_samples)], [(x_train, y_train), (x_test, y_test)]):

			for name in self._dtypes['numerical']:
				values = self._dataframe[[name]].iloc[self._index[indx]]
				x[name] = values

				if name in self._dtypes['targets']:
					output_name = f'classifier.{name}'
					y[output_name] = values
					
					gate_name = f'gate.{name}'
					x[gate_name] = self._dataframe[[gate_name]].iloc[self._index[indx]]

			for name in self._dtypes['categorical']:
				values = to_categorical(self._dataframe[[name]].iloc[self._index[indx]], num_classes=self._processors[name].n_tokens+1)[:,1:]
				x[name] = values

				if name in self._dtypes['targets']:
					output_name = f'classifier.{name}'
					y[output_name] = values

					gate_name = f'gate.{name}'
					x[gate_name] = self._dataframe[[gate_name]].iloc[self._index[indx]]

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
		early_stop = CustomEarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=0,
			mode='auto', baseline=None, restore_best_weights=True)
		self._model.classifier.fit(x=x_train, y=y_train, batch_size=batch_size, validation_data=(x_test, y_test),
			epochs=1000000, verbose=0, callbacks=[early_stop])

		return


	def _get_generator_batch(self, n, distribution='standard'):

		x = {}
		y = {}

		self._choice = np.random.choice(self._index,size=n, replace=True)
		dataframe = self._dataframe.iloc[self._choice]

		z_shape = (int(np.ceil(n/2)), self._model.n_latent,)

		if distribution == 'standard':
			z = tf.random.normal(z_shape, mean=0.0, stddev=1.0).numpy()

		elif distribution == 'truncated':
			z = tf.random.truncated_normal(z_shape, mean=0.0, stddev=1.0).numpy()
		
		z = np.vstack([z,-z])
		np.random.shuffle(z)

		x['z'] = z[:n]

		x.update({name:dataframe[[name]].values for name in set(dataframe.columns) - self._dtypes['categorical']})
		x.update({name:to_categorical(dataframe[name], num_classes=self._processors[name].n_tokens+1)[:,1:] for name in self._dtypes['categorical']})

		y.update({f'gan.classifier.{name}':x[name] for name in self._dtypes['targets']})

		discrimination = np.ones(shape=(n,1))
		y['gan.discriminator.discrimination.real'] = discrimination
		y['gan.discriminator.discrimination.synthetic'] = 1.5 - discrimination
		y['gan.discriminator.discrimination'] = discrimination

		return x, y
		
	
	def _get_synthetic_samples(self, n, distribution='standard', class_labels=None):
		
		x = {}
		y = {}

		latent_x, _  = self._get_generator_batch(n, distribution=distribution)
		x.update({name:latent_x[name] for name in self._model.discriminator.input_names if name in latent_x})

		generator_input = {name:latent_x[name] for name in self._model.generator.input_names}

		if class_labels is not None:
			generator_input.update({name:values.to_numpy().reshape(-1,1) for name,values in class_labels.items() if name in self._dtypes['numerical']})
			generator_input.update({name:to_categorical(values, num_classes=self._processors[name].n_tokens+1)[:,1:] for name,values in class_labels.items() if name in self._dtypes['categorical']})
			x.update(generator_input)

		x.update(self._model.generator(generator_input))

		discrimination = np.ones(shape=(n,1))
		y['discriminator.discrimination.real'] = 1.5 - discrimination
		y['discriminator.discrimination.synthetic'] = discrimination
		y['discriminator.discrimination'] = 1 - discrimination

		return x, y


	def _get_real_samples(self, n):

		x = {}
		y = {}

		self._choice = np.random.choice(self._index,size=n, replace=True)
		dataframe = self._dataframe.iloc[self._choice]

		x.update({name:dataframe[[name]].values for name in self._dtypes['numerical']})
		x.update({name:to_categorical(dataframe[name], num_classes=self._processors[name].n_tokens+1)[:,1:] for name in self._dtypes['categorical']})

		discrimination = np.ones(shape=(n,1))
		y['discriminator.discrimination.real'] = discrimination
		y['discriminator.discrimination.synthetic'] = 1.5 - discrimination
		y['discriminator.discrimination'] = discrimination

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

		loss_batch = []

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		x_generator, y_generator = self._get_generator_batch(batch_size)
		self._model.gan.train_on_batch(x_generator, y_generator)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)

		x_generator, y_generator = self._get_generator_batch(batch_size)
		loss = self._model.gan.train_on_batch(x_generator, y_generator)
		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		loss_batch.append(loss[0])

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		x_generator, y_generator = self._get_generator_batch(batch_size)
		loss = self._model.gan.train_on_batch(x_generator, y_generator)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		loss_batch.append(loss[0])

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		x_generator, y_generator = self._get_generator_batch(batch_size)
		loss = self._model.gan.train_on_batch(x_generator, y_generator)
		loss_batch.append(loss[0])

		x_generator, y_generator = self._get_generator_batch(batch_size)
		loss = self._model.gan.test_on_batch(x_generator, y_generator)
		loss_batch.append(loss[0])

		loss = np.mean(loss_batch)

		return loss


	def _train_epoch(self, batch_size, n_batches):

		loss_epoch = []

		for _ in range(n_batches):
			loss = self._train_batch(batch_size)
			loss_epoch.append(loss)

		loss = np.mean(loss_epoch)

		return loss


	def _get_dtypes(self, df, numerical_names, categorical_names, target_names):

		dtypes = {'numerical': none_to_set(numerical_names),
				  'categorical': none_to_set(categorical_names),
				  'features': set(),
				  'targets': none_to_set(target_names)}

		self._verify_columns(df, dtypes)

		for name, values in df.items():
			if name not in dtypes['numerical'] | dtypes['categorical']:
				if np.issubdtype(values.dtype, np.number):
					dtypes['numerical'].update({name})

				else:
					dtypes['categorical'].update({name})

			if name not in dtypes['targets']:
				dtypes['features'].update({name})

		return dtypes


	def _verify_columns(self, df, dtypes):

		for name in dtypes['targets']:
			if not name in df:
				raise ColumnNotFound(f'{name} is given as a target column name but is not found in the data.')

		for name in dtypes['numerical']:
			if not name in df:
				raise ColumnNotFound(f'{name} is given as a numerical column name but is not found in the data.')

		for name in dtypes['categorical']:
			if not name in df:
				raise ColumnNotFound(f'{name} is given as a categorical column name but is not found in the data.')

		return


	def _verify_class_labels(self, class_labels):

		for name in class_labels:
			if not name in self._dtypes['categorical'] | self._dtypes['targets']:
				raise ColumnNotFound(f"""{name} is given as a class label column name but is not a valid categorical \
							or target feature name. Valid categorical and target name are: {list(dtypes['categorical'] | dtypes['targets'])}""")

		return


	def _get_processors(self, df):

		processors = {}

		for name, values in df.items():
			if name in self._dtypes['numerical']:
				processor = NumericalProcessor(name, values)
				processors[name] = processor

			if name in self._dtypes['categorical']:
				processor = CategoricalProcessor(name, values)
				processors[name] = processor

		return processors


	def _apply_preprocess(self, series):

		name = series.name
		if name in self._processors:
			series = self._processors[name].preprocess(series)
		
		return series


	def _get_gates(self, df):

		for name in self._dtypes['numerical'] | self._dtypes['categorical']:
			gate_name = f'gate.{name}'
			df[gate_name] = df[name].notna().astype(int)

		return df


	def _preprocess(self, df):

		processed_df = df.apply(self._apply_preprocess)

		return processed_df


	def _setup_dataframe(self, df):

		df = self._get_gates(df)
		processed_df = self._preprocess(df)
		processed_df = processed_df.reset_index(drop=True)


		return processed_df


	def _to_dataframe(self, data):

		data_dict = {}

		for name in self._dtypes['numerical']:
			data_dict[name] = np.array(data[name]).flatten()

		for name in self._dtypes['categorical']:
			data_dict[name] = data[name].argmax(1)

		df = pd.DataFrame(data_dict)

		return df


	def _apply_postprocess(self, series):

		name = series.name
		if name in self._processors:
			series = self._processors[name].postprocess(series)
		
		return series


	def _apply_gates(self, df):

		for name in self._processors:
			gate_name = f'gate.{name}'
			gate = (self._dataframe[gate_name][self._choice]).reset_index(drop=True)
		
			if name in self._dtypes['numerical']:
				gate[gate == 0.] = np.nan
				df[name] = df[name] * gate

			else:
				df[name] = (df[name] + 1) * gate

		return df


	def _postprocess(self, df):

		processed_df = df.apply(self._apply_postprocess)

		return processed_df