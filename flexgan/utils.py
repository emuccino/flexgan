import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from .linearizer import Linearizer



def none_to_set(x):

	if not x:
		return set()
	else:
		return set(x)


class NumericalProcessor():

	def __init__(self, name, series):

		data = series.to_numpy(copy=True)

		self._name = name
		self._scaler = self._get_scaler(data)
		self._linearizer = self._get_linearizer(data)


	def _get_scaler(self, data):

		values = data[~np.isnan(data)]

		unbounded = ((values == values.min()).sum() == 1, (values == values.max()).sum() == 1)
		delta = 2 / len(values)

		scaler = MinMaxScaler(feature_range=((delta*unbounded[0])-1, 1-(delta*unbounded[1])), copy=True)
		scaler.fit(data.reshape(-1,1))

		return scaler


	def _get_linearizer(self, data):

		if 'int' in str(data.dtype).lower():
			rounded = 0
		else:
			rounded = None

		linearizer = Linearizer(rounded=rounded)
		linearizer.fit(data)

		return linearizer


	def preprocess(self, series):

		data = series.to_numpy()
		data = self._linearizer.transform(data)
		data = self._scaler.transform(data.reshape(-1,1))
		data = np.nan_to_num(data.flatten())
		series = pd.Series(data, name=self._name)

		return series


	def postprocess(self, series):
		
		data = series.to_numpy()
		data = self._scaler.inverse_transform(data.reshape(-1,1))
		data = self._linearizer.inverse_transform(data.flatten())
		series = pd.Series(data, name=self._name)

		return series



class CategoricalProcessor():

	def __init__(self, name, series):

		self._name = name
		self._dtype = series.dtype

		series = series.dropna().astype(str)
			
		self._tokenizer = self._get_tokenizer(series)
		self.n_tokens = len(self._tokenizer)


	def _get_tokenizer(self, series):
		
		tokenizer = pd.DataFrame(series, copy=True).groupby(by=self._name).count().reset_index()
		tokenizer['token'] = np.arange(len(tokenizer)) + 1
		tokenizer = tokenizer.append({self._name: '__FLEX_NAN__', 'token': 0}, ignore_index=True)

		return tokenizer


	def preprocess(self, series):

		series = pd.DataFrame(series, copy=True).fillna('__FLEX_NAN__').astype(str).merge(self._tokenizer, how='left').drop(columns=self._name).rename(columns={'token': self._name})[self._name]
		
		return series


	def postprocess(self, series):

		series = pd.DataFrame({'token':series}).merge(self._tokenizer, how='left')[self._name]
		series[series == '__FLEX_NAN__'] = np.nan
		series = series.astype(self._dtype)

		return series



class Error(Exception):
	pass

class ColumnNotFound(Error):
	pass



class CustomEarlyStopping(EarlyStopping):

	def __init__(self, **kwargs):
		
		super(CustomEarlyStopping, self).__init__(**kwargs)


	def on_train_begin(self, logs=None):
		# Allow instances to be re-used
		self.wait = 0
		self.stopped_epoch = 0
		if self.baseline is not None:
			self.best = self.baseline
		else:
			self.best = np.Inf if self.monitor_op == np.less else -np.Inf
		self.worst = np.Inf if self.monitor_op == np.greater else -np.Inf
		self.best_weights = None


	def on_epoch_end(self, epoch, logs=None):
		current = self.get_monitor_value(logs)
		if current is None:
			return

		if self.monitor_op(current - self.min_delta, self.best):
			self.best = current
			self.wait = 0
			if self.restore_best_weights:
				self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				if self.restore_best_weights:
					if self.verbose > 0:
						print('Restoring model weights from the end of the best epoch.')
					self.model.set_weights(self.best_weights)

		if self.monitor_op(self.worst, current + self.min_delta):
			self.best = np.Inf if self.monitor_op == np.less else -np.Inf
			self.worst = current
			self.wait = 0