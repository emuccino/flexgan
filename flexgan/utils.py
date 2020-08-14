import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from scipy.special import comb

from .linearizer import Linearizer

def none_to_set(x):

	if not x:
		return set()
	else:
		return set(x)

class NumericalProcesssor():

	def __init__(self, name, data):
		
		self.name = name
		self._scaler = self._get_scaler(data)
		self._linearizer = self._get_linearizer(data)


	def _get_scaler(self, data):

		values = data.values
		values = values[~np.isnan(values)]

		unbounded = ((values == values.min()).sum() == 1, (values == values.max()).sum() == 1)
		delta = 2 / len(values)

		scaler = MinMaxScaler(feature_range=((delta*unbounded[0])-1, 1-(delta*unbounded[1])), copy=True)

		return scaler


	def _get_linearizer(self, data):

		if 'int' in str(data.dtype).lower():
			rounded = 0
		else:
			rounded = None

		linearizer = Linearizer(rounded=rounded)

		return linearizer


	def preprocess(self, data):

		values = pd.DataFrame(data).values
		values = self._linearizer.fit_transform(values)
		values = self._scaler.fit_transform(values)
		values = np.nan_to_num(values)

		return values


	def postprocess(self, data, keep_index=False):
		
		values = pd.DataFrame(data)
		values = self._scaler.inverse_transform(data)
		values = self._linearizer.inverse_transform(values)

		return values



class CategoricalProcesssor():

	def __init__(self, name, data):

		self._name = name
		self._dtype = data.dtypes

		data = pd.DataFrame(data, columns=[name]).dropna(axis=0).astype(str)
			
		self._tokenizer = self._get_tokenizer(data)
		self.n_tokens = len(self._tokenizer)


	def _get_tokenizer(self, data):
		
		tokenizer = data.groupby(by=self._name).count().reset_index()
		tokenizer['token'] = np.arange(len(tokenizer)) + 1
		tokenizer = tokenizer.append({self._name: '__flex_nan__', 'token': 0}, ignore_index=True)

		return tokenizer


	def preprocess(self, data):

		data = pd.DataFrame(data).fillna('__flex_nan__').astype(str).merge(self._tokenizer, how='left').drop(columns=self._name).rename(columns={'token': self._name})
		
		return data.values


	def postprocess(self, data, keep_index=False):

		data = np.hstack([(data == 0.).all(1).reshape(-1,1).astype(int),data]).argmax(1).flatten()
		if keep_index:
			return data

		data = pd.DataFrame({'token':data}).merge(self._tokenizer, how='left').drop(columns='token')
		data[data == '__flex_nan__'] = np.nan
		data = data.astype(self._dtype)

		return data.values


class Error(Exception):
	pass

class ColumnNotFound(Error):
	pass