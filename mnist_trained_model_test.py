from matplotlib import pyplot as plt
import numpy as np
import bren as br


model = br.nn.models.load_model("mnist_train")

X, Y, X_test, Y_test = br.nn.datasets.load_mnist()

def one_hot_array(arr):
    unique = np.unique(arr)
    return br.Variable(np.eye(unique.size)[arr.flatten().astype("int")])

batch = X.numpy()[100:]
labels = Y.numpy()[100:]
one_hot_labels = one_hot_array(labels)

running = True

while running:
	randidx = input("Input an index between 0 and 9999 (both inclusive): ")
	if randidx == "exit":
		running = False
		continue
	else: randidx = int(randidx)

	predicted_values = model.predict([batch[randidx]])
	argmax_predicted = np.argmax(predicted_values)
	classes = np.unique(labels)

	print(np.squeeze(predicted_values), np.squeeze(one_hot_labels[randidx]))

	fig, axes = plt.subplots(nrows=2, ncols=1)

	colour = np.repeat("b", predicted_values.size)

	if argmax_predicted == labels[randidx]: colour[argmax_predicted] = "g"
	if argmax_predicted != labels[randidx]: 
			colour[int(argmax_predicted)] = "r"
			colour[int(labels[randidx])] = "g"

	# plot the data
	axes[0].imshow(batch[randidx])
	axes[1].bar(classes, np.squeeze(predicted_values), color=colour)

	plt.xticks(classes)
	plt.show()