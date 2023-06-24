import bren as br
import numpy as np


# X = br.Variable([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]], dtype="float16")[..., np.newaxis]
# Y = br.Variable([[[0]], [[1]], [[1]], [[0]]], dtype="float16")[..., np.newaxis]

# print(br.Variable([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]], dtype="float16")[..., np.newaxis].shape)
# print(br.Variable([[[0]], [[1]], [[1]], [[0]]], dtype="float16")[..., np.newaxis].shape)

#the model preprocess the data into batches in model...therefore only need 2d array...
X = br.Variable([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float16")
Y = br.Variable([0, 1, 1, 0], dtype="float16")


model = br.nn.models.Sequential([
    br.nn.layers.Dense(32, activation="relu"),
    br.nn.layers.Dense(32, activation="relu", use_bias=False),
    br.nn.layers.Dense(1),
])

model.assemble(
    loss="mse",
    optimiser=br.nn.optimisers.Adam(0.01),
    metrics=["accuracy"]
)

model.fit(X, Y, epochs=50, batch_size=1, shuffle=True)

model.save_weights("model.weights")

print(model.predict(X))