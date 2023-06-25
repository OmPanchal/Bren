import numpy as np
import bren as br
from bren.nn.metrics import get_metric
from bren.nn.losses import get_loss
from bren.nn.optimisers import get_optimiser
import pickle


# TODO: Try to impement steps into training


def set_metric(metric):
	out = None

	if type(metric) is str:
		out = get_metric(metric)()
	elif type(metric).__name__ == "function":
		out = get_metric(metric.__name__)()
	elif issubclass(type(metric), br.nn.metrics.Metric):
		out = metric
	else: out = get_metric(metric.__name__)()

	return out

def set_loss(loss):
	out = None

	if type(loss) is str:
		out = get_loss(loss)()
	elif type(loss).__name__ == "function":
		out = get_loss(loss.__name__)()
	elif issubclass(type(loss), br.nn.losses.Loss):
		out = loss
	else: out = get_loss(loss.__name__)()

	return out

def set_optimiser(optimiser):
	out = None

	if type(optimiser) is str:
		out = get_optimiser(optimiser)()
	elif issubclass(type(optimiser), br.nn.optimisers.Optimiser):
		out = optimiser
	else: out = get_optimiser(optimiser.__name__)()

	return out

class Model(object):
	def __init__(self, *args, **kwargs) -> None:
		self.training = False
		self.__assembled = False
		self.__built = False

		self.optimiser = None
		self.metrics = []
		self.loss = None

		self.trainable = kwargs.get("params") or []
		# self.predict = np.vectorize(self.predict, cache=True)

	@property
	def config(self): return self.__config
	# @config.setter
	# def config(self, val): print("nonono")

	def add_config(self, key, value): 
		print({**self.config, key: value})
		self.__config.update({**self.config, key: value})

	# actual functionality of the model...
	def call(self, x, training=None): ...

	def build(self, input): 
		self.__built = True
		self.call(input[0]) # run forward the network with the first value of the features to builc the weights layers
	
	# gets the different attributes such as optimiser 
	def assemble(self, loss=None, optimiser=None, metrics=None, **kwargs):
		self.__assembled = True
		self.loss = set_loss(loss)
		self.optimiser = set_optimiser(optimiser)
		self.metrics.append(set_metric(loss))

		for metric in metrics:
			self.metrics.append(set_metric(metric))

		# config is only established after the model has been assembled!
		self.__config = {
			"optimiser": self.optimiser.__class__.__name__,
			"loss": self.loss.__class__.__name__,
			"trainable": self.trainable,
			"model": self.__class__.__name__
		}
		
		self.__config["metrics"] = []
		for metric in self.metrics:
			self.__config["metrics"].append(metric.__class__.__name__)

	def fit(self, x, y, epochs=1, shuffle=False, batch_size=1, *args, **kwargs):
		if not self.__assembled: raise RuntimeError("The model should be assembled before you can train with it.")
		if not self.__built: self.build(x)
		
		X_batch = br.nn.preprocessing.split_uneven(x, batch_size)[..., np.newaxis]
		Y_batch = br.nn.preprocessing.split_uneven(y, batch_size)[..., np.newaxis]

		if shuffle: br.nn.preprocessing.shuffle(X_batch, Y_batch)

		for i in range(1, epochs + 1):
			self.__train_batch(X_batch, Y_batch)

			print(f"EPOCH {i}/{epochs}: ", end="")

			for metric in self.metrics:
				print(metric.__class__.__name__, ":", metric.result().numpy(), end=" - ")
				metric.reset()
			print()

	def add_weight(self, val, **kwargs):
		self.trainable.extend(val)

	def __forward_update(self, X, Y, training=None):
		loss = []
		for x, y in zip(X, Y):
			Z = self.call(x, training=training)
			loss.append(self.loss(Z, y))
			
			for metric in self.metrics:
				metric.update(Z, y)

		return np.sum(loss)

	def __train_batch(self, X_batches, Y_batches):
		for x, y in zip(X_batches, Y_batches):
			with br.Graph() as g:
				loss = self.__forward_update(x, y, training=True)
				grad = g.grad(loss, self.trainable)
				self.optimiser.apply_gradients(self.trainable, grad)
		
	def predict(self, X): 
		output = []
		for x in X:
			output.append(self.call(x[..., np.newaxis]).numpy())

		return np.array(output)
	
	def save(self, filepath): 
		with open(filepath, "wb") as f:
			pickle.dump(self.__config, f)
			

#! TRAIN
#* 	BATCH
#? 		FEATURES
#? 		FEATURES
#~	METRIC
#~	LOSS
#* 	BATCH
#? 		FEATURES
#? 		FEATURES
#~	METRIC
#~	LOSS


# FORWARD --> [p1, p2, p3,...] 

# (DATASET, BATCH, FEATURES, ...)