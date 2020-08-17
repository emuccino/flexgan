from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Layer,Input,Concatenate,Activation,Reshape,Multiply,LeakyReLU,Add
from tensorflow.keras.optimizers import Nadam,Adam,SGD
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, MSE, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.activations import sigmoid

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.utils import get_custom_objects



def clipped_loss(y_true,y_pred):

	mask = K.cast(K.argmax(y_true) != K.argmax(y_pred), dtype='float32')

	return K.mean(categorical_crossentropy(y_true,y_pred,from_logits=True) * mask)


class UniformNoise(Layer):
	def __init__(self, maxval):
		super(UniformNoise, self).__init__()

		self.maxval = maxval

	def call(self, inputs):
		random_noise = K.random_uniform(K.shape(inputs), minval=0., maxval=self.maxval)

		outputs = inputs + random_noise

		return outputs / K.sum(outputs,axis=-1, keepdims=True)


class GeneratorModel():

	def __init__(self, processors, dtypes, n_latent, n_neurons, n_layers, model_path):

		self._optimizer = Adam(0.0002)
		self.generator = self._compile_generator(processors, dtypes, n_latent, n_neurons, n_layers)
		self.discriminator = self._compile_discriminator(processors, dtypes, n_latent, n_neurons, n_layers)
		if dtypes['targets']:
			self.classifier = self._compile_classifier(processors, dtypes, n_latent, n_neurons, n_layers)
		self.gan = self._compile_gan(processors, dtypes, n_latent, n_neurons, n_layers)

		if model_path:
			self._model.gan.load_weights(model_path)


	def _compile_generator(self, processors, dtypes, n_latent, n_neurons, n_layers):
		
		model_name = 'generator'
		batchnorm = 'moderate'

		inputs = {}
		outputs = {}

		name = 'z'
		inputs[name] = Input(shape=(n_latent,), name=name)
		latent_net = inputs[name]

		categorical_nets = []

		for name in dtypes['categorical']:
			n_tokens = processors[name].n_tokens
			inputs[name] = Input(shape=(n_tokens,), name=name)
			net = inputs[name]
			categorical_nets.append(net)

		for name in dtypes['numerical'] & dtypes['targets']:
			inputs[name] = Input(shape=(1,), name=name)
			net = inputs[name]
			categorical_nets.append(net)

		if categorical_nets:
			net = self._mask_net(latent_net, categorical_nets, n_neurons, batchnorm=batchnorm)
		else:
			net = latent_net

		for _ in range(n_layers):
			net = self._dense(net, n_neurons, activation='LeakyReLU', kernel_initializer='he_normal', batchnorm=batchnorm)

		for name in dtypes['numerical'] & dtypes['features']:
			outputs[name] = Dense(1, activation='tanh', kernel_initializer='glorot_normal', name=name)(net)
		
		generator = Model(inputs=inputs, outputs=outputs, name=model_name)

		return generator

	
	# def _compile_generator(self, processors, dtypes, n_latent, n_neurons, n_layers):
		
	# 	model_name = 'generator'
	# 	batchnorm = 'moderate'

	# 	inputs = {}
	# 	outputs = {}

	# 	name = 'z'
	# 	inputs[name] = Input(shape=(n_latent,), name=name)
	# 	latent_net = inputs[name]

	# 	categorical_nets_base = []

	# 	for name in dtypes['categorical']:
	# 		n_tokens = processors[name].n_tokens
	# 		inputs[name] = Input(shape=(n_tokens,), name=name)
	# 		net = inputs[name]
	# 		categorical_nets_base.append(net)

	# 	for name in dtypes['numerical'] & dtypes['targets']:
	# 		inputs[name] = Input(shape=(1,), name=name)
	# 		net = inputs[name]
	# 		categorical_nets_base.append(net)

	# 	for name in dtypes['numerical'] & dtypes['features']:
	# 		if categorical_nets_base:
	# 			net = self._mask_net(latent_net, categorical_nets_base, n_neurons, batchnorm=batchnorm)
	# 		else:
	# 			net = latent_net

	# 		for _ in range(n_layers):
	# 			net = self._dense(net, n_neurons, activation='LeakyReLU', kernel_initializer='he_normal', batchnorm=batchnorm)

	# 		outputs[name] = Dense(1, activation='tanh', kernel_initializer='glorot_normal', name=name)(net)
		
	# 	generator = Model(inputs=inputs, outputs=outputs, name=model_name)

	# 	return generator



	def _compile_discriminator(self, processors, dtypes, n_latent, n_neurons, n_layers):

		model_name = 'discriminator'
		batchnorm = 'slow'

		inputs = {}
		outputs = {}
		loss = {}
		
		numerical_nets_base = []
		categorical_nets_base = []

		for name in dtypes['numerical']:
			inputs[name] = Input(shape=(1,), name=name)
			net = inputs[name]
			numerical_nets_base.append(net)

		for name in dtypes['categorical']:
			n_tokens = processors[name].n_tokens
			inputs[name] = Input(shape=(n_tokens,), name=name)
			net = inputs[name]
			net = UniformNoise(0.2/n_tokens)(net)
			categorical_nets_base.append(net)

		if len(numerical_nets_base) > 1:
			numerical_net_base = Concatenate()(numerical_nets_base)
		else:
			numerical_net_base = numerical_nets_base[0]

		for output_type in ['discrimination.real', 'discrimination.synthetic', 'discrimination']:
			if categorical_nets_base:
				net = self._mask_net(numerical_net_base, categorical_nets_base, n_neurons, batchnorm=batchnorm)
			else:
				net = numerical_net_base

			for _ in range(n_layers):
				net = self._dense(net, n_neurons, activation='LeakyReLU', kernel_initializer='he_normal', batchnorm=batchnorm)

			output_name = f'{model_name}.{output_type}'
			if output_type == 'discrimination':
				outputs[output_name] = Dense(1, kernel_initializer='he_normal', name=output_name)(net)
			else:
				outputs[output_name] = Dense(1, activation='relu', kernel_initializer='he_normal', name=output_name)(net)
			loss[output_name] = BinaryCrossentropy(from_logits=True)

		discriminator = Model(inputs=inputs, outputs=outputs, name=model_name)
		discriminator.compile(loss=loss, optimizer=self._optimizer)

		return discriminator
	

	def _compile_classifier(self, processors, dtypes, n_latent, n_neurons, n_layers):

		model_name = 'classifier'
		batchnorm = True

		inputs = {}
		outputs = {}
		loss = {}
		numerical_nets_base = {}
		categorical_nets_base = {}
		output_nets = []

		for name in dtypes['numerical']:
			inputs[name] = Input(shape=(1,), name=name)
			numerical_nets_base[name] = inputs[name]

		for name in dtypes['categorical']:
			n_tokens = processors[name].n_tokens
			inputs[name] = Input(shape=(n_tokens,), name=name)
			categorical_nets_base[name] = UniformNoise(0.2/n_tokens)(inputs[name])

		for name in dtypes['targets']:
			gate_name = f'gate.{name}'
			inputs[gate_name] = Input(shape=(1,), name=gate_name)

		for target in dtypes['targets']:
			numerical_nets = [net for name, net in numerical_nets_base.items() if name != target]
			categorical_nets = [net for name, net in categorical_nets_base.items() if name != target]

			if len(numerical_nets) > 1:
				numerical_net = Concatenate()(numerical_nets)
			else:
				numerical_net = numerical_nets[0]

			if categorical_nets:
				net = self._mask_net(numerical_net, categorical_nets, n_neurons, batchnorm=batchnorm)
			else:
				net = numerical_net

			for _ in range(n_layers):
				net = self._dense(net, n_neurons, activation='LeakyReLU', kernel_initializer='he_normal', batchnorm=batchnorm)

			head = net

			gate_name = f'gate.{target}'
			output_name = f'{model_name}.{target}'
			if target in dtypes['categorical']:
				net = Dense(processors[name].n_tokens, kernel_initializer='glorot_normal')(head) * inputs[gate_name]
				outputs[output_name] = Lambda(lambda x: x, name = output_name, output_shape=(processors[name].n_tokens,))(net)
				loss[output_name] = CategoricalCrossentropy(from_logits=True)

			else:
				net = Dense(1, kernel_initializer='glorot_normal')(head) * inputs[gate_name]
				outputs[output_name] = Lambda(lambda x: x, name = output_name, output_shape=(1,))(net)
				loss[output_name] = 'mse'

		classifier = Model(inputs=inputs, outputs=outputs, name=model_name)
		classifier.compile(loss=loss, optimizer=Nadam())

		return classifier
		

	def _compile_gan(self, processors, dtypes, n_latent, n_neurons, n_layers):
		
		model_name = 'gan'

		inputs = {}
		generator_inputs = {}
		discriminator_inputs = {}
		classifier_inputs = {}
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

			if name in dtypes['targets']:
				gate_name = f'gate.{name}'
				inputs[gate_name] = Input(shape=(1,), name=gate_name)

		for name in self.generator.input_names:
			generator_inputs[name] = inputs[name]

		for name in self.discriminator.input_names:
			if name not in dtypes['numerical'] & dtypes['features']:
				discriminator_inputs[name] = inputs[name]

		if dtypes['targets']:
			for name in self.classifier.input_names:
				if name not in dtypes['numerical'] & dtypes['features']:
					classifier_inputs[name] = inputs[name]

		generator_outputs = self.generator(generator_inputs)

		for name in dtypes['numerical'] & dtypes['features']:
			gate_name = f'gate.{name}'
			gate = inputs[gate_name]
			net = generator_outputs[name]

			net = net * gate
			net = Lambda(lambda x: x, output_shape=(1,))(net)

			discriminator_inputs[name] = net
			classifier_inputs[name] = net

		self.discriminator.trainable = False
		discriminator_outputs = self.discriminator(discriminator_inputs)

		for name, output in discriminator_outputs.items():
			gan_name = f'{model_name}.{name}'
			outputs[gan_name] = Lambda(lambda x: x, name = gan_name, output_shape=output.shape)(output)
			loss[gan_name] = self.discriminator.loss[name]

		if dtypes['targets']:
			self.classifier.trainable = False
			classifier_outputs = self.classifier(classifier_inputs)

			for name,output in classifier_outputs.items():
				gan_name = f'{model_name}.{name}'
				outputs[gan_name] = Lambda(lambda x: x, name = gan_name, output_shape=output.shape)(output)
				loss[gan_name] = self.classifier.loss[name]

			# for name in dtypes['categorical'] & dtypes['targets']:
			# 	loss[f'{model_name}.classifier.{name}'] = clipped_loss

		gan = Model(inputs=inputs, outputs=outputs, name=model_name)
		gan.compile(loss=loss, optimizer=self._optimizer)

		return gan


	def _dense(self, net, n, activation=None, kernel_initializer='glorot_uniform', batchnorm=None, dropout=None):

		if activation == 'LeakyReLU':
			net = Dense(n, activation=None, kernel_initializer=kernel_initializer)(net)
			net = LeakyReLU()(net)

		else:
			net = Dense(n, activation=activation, kernel_initializer=kernel_initializer)(net)

		if batchnorm:
			if batchnorm == 'slow':
				net = BatchNormalization()(net)
				# net = BatchNormalization(renorm=True, momentum=0.5, renorm_momentum=0.5)(net)
			elif batchnorm == 'moderate':
				net = BatchNormalization()(net)
				# net = BatchNormalization(renorm=True, momentum=0.9, renorm_momentum=0.9)(net)
			else:
				net = BatchNormalization()(net)

		if dropout:
			net = Dropout(dropout)(net)

		return net


	def _mask_net(self, numerical_net, categorical_nets, n, batchnorm=None):

		categorical_nets = [self._dense(net, n, activation='softmax', kernel_initializer='glorot_normal')
			for net in categorical_nets]

		if len(categorical_nets) > 1:
			categorical_net = Add()(categorical_nets)
		else:
			categorical_net = categorical_nets[0]

		numerical_net = self._dense(numerical_net, n, activation=None, kernel_initializer='he_normal')
		net = numerical_net * categorical_net

		if batchnorm:
			if batchnorm == 'slow':
				net = BatchNormalization()(net)
				# net = BatchNormalization(renorm=True, momentum=0.5, renorm_momentum=0.5)(net)
			elif batchnorm == 'moderate':
				net = BatchNormalization()(net)
				# net = BatchNormalization(renorm=True, momentum=0.9, renorm_momentum=0.9)(net)
			else:
				net = BatchNormalization()(net)

		return net