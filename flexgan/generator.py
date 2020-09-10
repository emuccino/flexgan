import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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
			numerical_names (list): List of numerical feature names. Specifying numerical names can help ensure that
				data is processed appropriately, however it is not required.
			categorical_names (list): List of categorical feature names. Specifying numerical names can help ensure that
				data is processed appropriately, however it is not required.
			resources (int): An value representing the model size and capacity. Larger values generally produce better results
				but lead to greater memory usage and slower training time. The default is 2.
			model_path (str): Path to load pretrained data generator model.

		Returns:
			DataGenerator: Initalized class object for synthetic data generation.
				Call .train() to begin training the synthetic data generator model.
				Call .generate_data() to generate synthetic data.
				Call .save_model() to save a trained synthetic data generator model for future use.
		"""

	return DataGenerator(*args, **kwargs)


class DataGenerator():

	def __init__(self, dataframe=None, numerical_names=None, categorical_names=None, csv_path=None, resources=2, model_path=None):

		if csv_path:
			dataframe = pd.read_csv(csv_path)
		else:
			dataframe = pd.DataFrame(dataframe, copy=True).reset_index(drop=True)

		self._original_columns, self._columns, dataframe = self._get_column_names(dataframe)
		self._n_samples = len(dataframe)
		self._index = np.arange(self._n_samples)

		self._dtypes = self._get_dtypes(dataframe, numerical_names, categorical_names)
		self._processors = self._get_processors(dataframe)
		self._dataframe = self._setup_dataframe(dataframe)

		self._p, self._categorical_index, self._outer_index = self._get_sampling()

		self._model = FlexGANModel(self._processors, self._dtypes, resources, model_path)

		return


	def _get_column_names(self, df):

		original_columns = df.columns
		df = df.rename(columns={name:str(name) for name in df.columns})
		columns = df.columns

		return original_columns, columns, df


	def train(self, batch_size=1024, patience=200, model_path=None, checkpoint=False):

		"""
		Function for training synthetic data generator model.

		Args:
			batch_size (int): The number of data samples to use in each training batch.
				Larger batch_size generally produce better results but use more memory. The default is 1024.
			patience (int): The number of training epochs without improvement to allow before halting training.
				Larger patience generally produces better results but increases model training time. The default is 200.
			path (str): Optional path for saving the synthetic data generator model. E.g. 'path/my_flexgan_model.h5'
			checkpoint (bool): Option for saving periodic checkpoints. Only valid if model_path is provided.

		Returns:
			self
		"""

		batch_size = int(batch_size)
		patience = int(patience)
		n_batches = int(np.ceil(self._n_samples / (batch_size * 4)))

		history = []

		best_loss = np.inf
		worst_loss = -np.inf
		min_delta = np.inf

		base_learning_rate = 0.01

		for _ in range(2):
			loss = self._train_epoch(batch_size, n_batches)
			history.append(loss)

		while True:
			wait = 0
			epoch = 0

			K.set_value(self._model.gan.optimizer.learning_rate, base_learning_rate)

			stop = True
			while True:
				loss = self._train_epoch(batch_size, n_batches)
				history.append(loss)
				epoch += 1

				delta = (history[-2] - history[-3]) * (history[-1] - history[-2])

				if loss < best_loss:
					weights = self._model.gan.get_weights()

				if (loss < best_loss * 0.99) or (loss > worst_loss) or (delta < min_delta):
					if loss < best_loss * 0.99:
						best_loss = loss

						if model_path and checkpoint:
							self.save_model(model_path)

					if loss > worst_loss:
						worst_loss = loss
						best_loss = np.inf

					if delta < min_delta:
						min_delta = delta
						best_loss = np.inf

					wait = 0
					stop = False

				else:
					wait += 1

				if wait == patience:
					break
					
			if stop:
				self._model.gan.set_weights(weights)
				if model_path:
					self.save_model(model_path)

				return self

			learning_rate_schedule = np.exp(np.linspace(np.log(base_learning_rate),np.log(base_learning_rate/2),patience))

			for epoch in range(patience):
				learning_rate = learning_rate_schedule[epoch]
				K.set_value(self._model.gan.optimizer.learning_rate, learning_rate)

				loss = self._train_epoch(batch_size, n_batches)
				history.append(loss)

				delta = (history[-2] - history[-3]) * (history[-1] - history[-2])

				if (loss < best_loss) or (loss > worst_loss) or (delta < min_delta):
					if loss < best_loss:
						weights = self._model.gan.get_weights()
						best_loss = loss
						if model_path and checkpoint:
							self.save_model(model_path)

					if loss > worst_loss:
						worst_loss = loss
						best_loss = np.inf

					if delta < min_delta:
						min_delta = delta
						best_loss = np.inf

			base_learning_rate = base_learning_rate/2


	def generate_data(self, n_samples=None, class_labels=None, to_csv=None):

		"""
		Function for generating synthetic data.

		Args:
			n_samples (int): The number of data samples to generate. If None, generated sample count will be the
				equal to the original data sample count, or if. n_samples will be ignored if class_labels are supplied.
				samples as is in the original data. 
			class_labels (dict): A dictionary-like object (e.g. pandas.DataFrame) that contains categorical labels
				for custom class label distributions within generated data. If None, categorical labels are randomly
				drawn from the original class label distribution.
			to_csv (str): Path and filename to save synthetic data as csv. (optional)

		Returns:
			pandas.DataFrame: A dataframe cointaining synthetic data.
		"""

		if class_labels is not None:
			class_labels = pd.DataFrame(class_labels, copy=True)
			self._verify_class_labels(class_labels)
			class_labels = self._preprocess(class_labels)
			n_samples = len(class_labels)

		elif not n_samples:
			n_samples = self._n_samples

		data, _ = self._get_synthetic_samples(n_samples, class_labels=class_labels, distribution='truncated')

		df = self._to_dataframe(data)
		df = self._apply_gates(df)
		df = self._postprocess(df)
		df = df[self._columns].rename(columns={str(name):name for name in self._original_columns})

		if to_csv:
			df.to_csv(to_csv, index=False)

		return df


	def save_model(self, model_path):
		
		"""
		Function for saving a trained synthetic data generator model for future use.

		Args:
			model_path (str): Path and file name to save the synthetic data generator model. E.g. 'path/my_flexgan_model.h5'

		Returns:
			None
		"""

		self._model.gan.save_weights(model_path)

		return


	def _interpolate(self, data_list, n):

		p = self._p[self._choice]
		categorical_index = self._categorical_index[self._choice]
		outer_index = self._outer_index.iloc[self._choice]

		indx = np.arange(n)
		interp_indx = np.zeros(n, dtype=int)

		for c in set(categorical_index):
			c_indx = categorical_index == c
			i_indx = c_indx | outer_index[c]
			options = indx[i_indx]

			interp_indx[c_indx] = np.random.choice(options, size=sum(c_indx), replace=False, p=p[i_indx]/p[i_indx].sum())

		indxs = np.stack([indx, interp_indx], axis=-1)
		betas = np.expand_dims(np.random.dirichlet((1., 1.), size=n), axis=-1)

		interpolated_data = [{name: (betas * values[indxs]).sum(1) for name, values in data.items()} for data in data_list]

		return interpolated_data


	def _get_generator_batch(self, n, distribution='standard', p=None, interpolate=False):

		x = {}
		y = {}

		self._choice = np.hstack([np.random.choice(self._index, size=self._n_samples, replace=False, p=p)
							for _ in range(n//self._n_samples)] + [np.random.choice(self._index, size=n%self._n_samples, replace=False, p=p)])

		df = self._dataframe.iloc[self._choice]

		z_shape = (int(np.ceil(n/2)), self._model.n_latent,)

		if distribution == 'standard':
			z = tf.random.normal(z_shape, mean=0.0, stddev=1.0, dtype='float32').numpy()

		elif distribution == 'truncated':
			z = tf.random.truncated_normal(z_shape, mean=0.0, stddev=1.0, dtype='float32').numpy()
		
		z = np.vstack([z,-z])
		np.random.shuffle(z)

		x['z'] = z[:n]

		x.update({name:df[[name]].values for name in set(df.columns) - self._dtypes['categorical']})
		x.update({name:to_categorical(df[name], num_classes=self._processors[name].n_tokens+1)[:,1:] for name in self._dtypes['categorical']})

		y['gan.discriminator.discrimination'] = np.ones(shape=(n,1))

		if interpolate:
			x, y = self._interpolate([x, y], n)		

		return x, y
		
	
	def _get_synthetic_samples(self, n, distribution='standard', class_labels=None, p=None):
		
		x = {}
		y = {}

		latent_x, _  = self._get_generator_batch(n, distribution=distribution, p=p)
		x.update({name:latent_x[name] for name in self._model.discriminator.input_names if name in latent_x})

		generator_input = {name:latent_x[name] for name in self._model.generator.input_names}

		if class_labels is not None:
			generator_input.update({name:values.to_numpy().reshape(-1,1) for name,values in class_labels.items() if name in self._dtypes['numerical']})
			generator_input.update({name:to_categorical(values, num_classes=self._processors[name].n_tokens+1)[:,1:] for name,values in class_labels.items() if name in self._dtypes['categorical']})
			x.update(generator_input)

		x.update({name:values.numpy() for name, values in self._model.generator(generator_input).items()})

		y['discriminator.discrimination'] = np.zeros(shape=(n,1))

		return x, y


	def _get_real_samples(self, n):

		x = {}
		y = {}

		df = self._dataframe.iloc[self._choice]

		x.update({name:df[[name]].values for name in self._dtypes['numerical']})
		x.update({name:to_categorical(df[name], num_classes=self._processors[name].n_tokens+1)[:,1:] for name in self._dtypes['categorical']})

		y['discriminator.discrimination'] = np.ones(shape=(n,1))

		return x, y


	def _get_discriminator_batch(self, batch_size, p=None):

		x = {}
		y = {}

		half_batch = batch_size//2

		x_synthetic, y_synthetic = self._get_synthetic_samples(half_batch, p=p)
		x_real, y_real = self._get_real_samples(half_batch)

		x_synthetic, y_synthetic, x_real, y_real = self._interpolate([x_synthetic, y_synthetic, x_real, y_real],
			half_batch)

		for name in x_real.keys():
			x[name] = np.vstack([x_synthetic[name], x_real[name]])
		for name in y_real.keys():
			y[name] = np.vstack([y_synthetic[name], y_real[name]])

		return x, y


	def _train_batch(self, batch_size):

		p = self._p

		loss_batch = []

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size, p=p)
		x_generator, y_generator = self._get_generator_batch(batch_size, p=p, interpolate=True)
		self._model.gan.train_on_batch(x_generator, y_generator)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)

		x_generator, y_generator = self._get_generator_batch(batch_size, p=p, interpolate=True)
		loss = self._model.gan.train_on_batch(x_generator, y_generator)
		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size, p=p)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		loss_batch.append(loss)

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size, p=p)
		x_generator, y_generator = self._get_generator_batch(batch_size, p=p, interpolate=True)
		loss = self._model.gan.train_on_batch(x_generator, y_generator)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		loss_batch.append(loss)

		x_discriminator, y_discriminator = self._get_discriminator_batch(batch_size, p=p)
		self._model.discriminator.train_on_batch(x_discriminator, y_discriminator)
		x_generator, y_generator = self._get_generator_batch(batch_size, p=p, interpolate=True)
		loss = self._model.gan.train_on_batch(x_generator, y_generator)
		loss_batch.append(loss)

		x_generator, y_generator = self._get_generator_batch(batch_size, p=p)
		loss = self._model.gan.test_on_batch(x_generator, y_generator)
		loss_batch.append(loss)

		loss = np.mean(loss_batch)

		return loss


	def _train_epoch(self, batch_size, n_batches):

		loss_epoch = []

		for _ in range(n_batches):
			loss = self._train_batch(batch_size)
			loss_epoch.append(loss)

		loss = np.mean(loss_epoch)

		return loss


	def _get_dtypes(self, df, numerical_names, categorical_names):

		dtypes = {'numerical': none_to_set(numerical_names),
				  'categorical': none_to_set(categorical_names)}

		self._verify_columns(df, dtypes)

		for name, values in df.items():
			if name not in dtypes['numerical'] | dtypes['categorical']:
				if np.issubdtype(values.dtype, np.number):
					dtypes['numerical'].update({name})

				else:
					dtypes['categorical'].update({name})

		return dtypes


	def _verify_columns(self, df, dtypes):

		for name in dtypes['numerical']:
			if not name in df:
				raise ColumnNotFound(f'{name} is given as a numerical column name but is not found in the data.')

		for name in dtypes['categorical']:
			if not name in df:
				raise ColumnNotFound(f'{name} is given as a categorical column name but is not found in the data.')

		return


	def _verify_class_labels(self, class_labels):

		for name in class_labels:
			if not name in self._dtypes['categorical']:
				raise ColumnNotFound(f"""{name} is given as a class label column name but is not a valid categorical.\nValid categorical names are: {list(self._dtypes['categorical'])}""")

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


	def _get_sampling(self):

		if self._dtypes['categorical']:
			df = self._dataframe.copy()

			df['count'] = 1
			p = df.drop(columns=['count']).merge((1 / np.sqrt(df.groupby(list(self._dtypes['categorical'])).count()['count'])).reset_index(), how='left')['count'].values
			p = p/p.sum()

			categories_df = df[list(self._dtypes['categorical'])].drop_duplicates()
			categories_df['__flexgan_categories__'] = np.arange(len(categories_df))

			df = df.merge(categories_df, how='left')[list(self._dtypes['categorical'])+['__flexgan_categories__']]

			categorical_index = df['__flexgan_categories__'].values

			outer_index = pd.DataFrame({c:(df[df['__flexgan_categories__'] == c].iloc[0] != df).all(1).values for c in df['__flexgan_categories__']})

		else:
			p = np.ones(self._n_samples)
			p = p/p.sum()

			categorical_index = np.zeros(self._n_samples)

			outer_index = pd.DataFrame({0:np.full(self._n_samples, True)})

		return p, categorical_index, outer_index