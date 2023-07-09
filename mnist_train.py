import numpy as np
import bren as br


def one_hot_array(arr):
    unique = np.unique(arr)
    return br.Variable(np.eye(unique.size)[arr.flatten().astype("int")])


X, Y, X_test, Y_test = br.nn.datasets.load_mnist(dtype="float64")

Y = one_hot_array(Y)


model = br.nn.models.Sequential(layers=[
    br.nn.layers.Flatten(),
    br.nn.layers.Dense(64, activation="tanh"),
    br.nn.layers.Dense(128, activation="tanh"),
    br.nn.layers.Dense(256, activation="tanh"),
    br.nn.layers.Dense(10, activation=br.nn.layers.Softmax())
])

model.assemble(
    loss=br.nn.losses.CategoricalCrossEntropy,
    metrics=[br.nn.metrics.Accuracy],
    optimiser=br.nn.optimisers.Adam(learning_rate=0.001)
)

model.fit(X[0:1000], Y[0:1000], epochs=100, batch_size=10)

pred = model.predict(X[0:20])

for i, p in enumerate(pred): 
    print("PRED:", np.argmax(p), "ACT:", Y[i].numpy())

model.save("mnist_train")