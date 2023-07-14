import bren as br 
import numpy as np


X = br.Variable([[1, 1], [1, 0], [0, 1], [0, 0]], dtype="float64")
Y = br.Variable([0, 1, 1, 0], dtype="float64")


def belu(x): return x

def random(shape, **kwargs): 
	return br.Variable(np.random.random(size=shape))

model = br.nn.models.Sequential([
	br.nn.layers.Dense(32, activation=belu, weights_initialiser=random),
	br.nn.layers.Dense(32, activation="elu"),
	br.nn.layers.Dense(1, activation="elu")
])

model.assemble(
	loss="mse",
	optimiser=br.nn.optimisers.Adam(learning_rate=0.01),
	metrics=["accuracy"]
)

model.fit(X, Y, epochs=1000, batch_size=1, shuffle=True)

out = model.predict(X)
print(out)

model.save("test_model")
