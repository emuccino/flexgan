from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Multiply, Input, Concatenate, LeakyReLU, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


class FlexGANModel():

	def __init__(self, processors, dtypes, resources, model_path):

		n_latent = int(16 * resources)
		n_neurons = int(64 * resources)
		n_layers = int(1 + resources)

		self.n_latent = n_latent
		self._optimizer = SGD()

		self.generator = self._compile_generator(processors, dtypes, n_latent, n_neurons, n_layers)
		self.discriminator = self._compile_discriminator(processors, dtypes, n_latent, n_neurons, n_layers)
		self.gan = self._compile_gan(processors, dtypes, n_latent, n_neurons, n_layers)

		if model_path:
			self.gan.load_weights(model_path)


	def _compile_generator(self, processors, dtypes, n_latent, n_neurons, n_layers):

		model_name = 'generator'

		inputs = {}
		outputs = {}

		name = 'z'
		inputs[name] = Input(shape=(n_latent,), name=name)
		latent_net = inputs[name]

		categorical_nets = {}

		for name in dtypes['categorical']:
			inputs[name] = Input(shape=(processors[name].n_tokens,), name=name)
			categorical_nets[name] = inputs[name]

		if categorical_nets:
			mask_layers = self._get_mask_layers(n_neurons, categorical_nets)
		else:
			mask_net = None

		net = latent_net

		for _ in range(n_layers):
			if categorical_nets:
				mask_net = self._get_mask_net(mask_layers, categorical_nets, processors)

			net = self._dense(net, n_neurons, mask=mask_net)
	
		for name in dtypes['numerical']:
			outputs[name] = Dense(1, activation='tanh', kernel_initializer='he_normal', name=name)(net)
		
		generator = Model(inputs=inputs, outputs=outputs, name=model_name)

		return generator


	def _compile_discriminator(self, processors, dtypes, n_latent, n_neurons, n_layers):

		model_name = 'discriminator'

		inputs = {}
		outputs = {}
		loss = {}
		
		numerical_nets = []
		categorical_nets = {}

		for name in dtypes['numerical']:
			inputs[name] = Input(shape=(1,), name=name)
			numerical_nets.append(inputs[name])

		for name in dtypes['categorical']:
			inputs[name] = Input(shape=(processors[name].n_tokens,), name=name)
			categorical_nets[name] = inputs[name]

		if len(numerical_nets) > 1:
			numerical_net = Concatenate()(numerical_nets)
		else:
			numerical_net = numerical_nets[0]
		
		if categorical_nets:
			mask_layers = self._get_mask_layers(n_neurons, categorical_nets)
		else:
			mask_net = None

		net = numerical_net

		for _ in range(n_layers):
			if categorical_nets:
				mask_net = self._get_mask_net(mask_layers, categorical_nets, processors)

			net = self._dense(net, n_neurons, mask=mask_net)

		output_name = f'{model_name}.discrimination'
		outputs[output_name] = Dense(1, kernel_initializer='he_normal', name=output_name)(net)

		loss[output_name] = BinaryCrossentropy(from_logits=True)

		discriminator = Model(inputs=inputs, outputs=outputs, name=model_name)
		discriminator.compile(loss=loss, optimizer=self._optimizer)

		return discriminator
	

	def _compile_gan(self, processors, dtypes, n_latent, n_neurons, n_layers):
		
		model_name = 'gan'

		inputs = {}
		generator_inputs = {}
		discriminator_inputs = {}
		outputs = {}
		loss = {}

		name = 'z'
		inputs[name] = Input(shape=(n_latent,), name=name)
		generator_inputs[name] = inputs[name]

		for name in dtypes['numerical']:
			inputs[name] = Input(shape=(1,), name=name)
			
			gate_name = f'gate.{name}'
			inputs[gate_name] = Input(shape=(1,), name=gate_name)

		for name in dtypes['categorical']:
			inputs[name] = Input(shape=(processors[name].n_tokens,), name=name)

		for name in self.generator.input_names:
			generator_inputs[name] = inputs[name]

		for name in self.discriminator.input_names:
			if name not in dtypes['numerical']:
				discriminator_inputs[name] = inputs[name]

		generator_outputs = self.generator(generator_inputs)

		for name in dtypes['numerical']:
			gate_name = f'gate.{name}'
			gate = inputs[gate_name]
			net = generator_outputs[name]

			net = net * gate
			net = Lambda(lambda x: x, output_shape=(1,))(net)

			discriminator_inputs[name] = net

		self.discriminator.trainable = False
		discriminator_outputs = self.discriminator(discriminator_inputs, training=False)

		for name, output in discriminator_outputs.items():
			gan_name = f'{model_name}.{name}'
			outputs[gan_name] = Lambda(lambda x: x, name = gan_name, output_shape=output.shape)(output)
			loss[gan_name] = self.discriminator.loss[name]

		gan = Model(inputs=inputs, outputs=outputs, name=model_name)
		gan.compile(loss=loss, optimizer=self._optimizer)

		return gan


	def _dense(self, net, n, mask=None):

		net = Dense(n, activation=None, kernel_initializer='he_normal', use_bias=False)(net)
		net = LeakyReLU(0.1)(net)
		net = BatchNormalization(center=True, scale=False)(net)

		if mask != None:
			net = net * mask

		return net


	def _get_mask_layers(self, n, categorical_nets):

		mask_layers = {name: Dense(n, activation='sigmoid', kernel_initializer='he_normal', use_bias=False) for name in categorical_nets.keys()}

		return mask_layers


	def _get_mask_net(self, mask_layers, categorical_nets, processors):

		mask_nets = []

		for name, net in categorical_nets.items():
			procesor = processors[name]
			mask_net = mask_layers[name](net)
			mask_nets.append(mask_net)

		if len(mask_nets) > 1:
			mask_net = Multiply()(mask_nets)
		else:
			mask_net = mask_nets[0]

		return mask_net