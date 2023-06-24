import pickle as pk


with open("model", "rb") as f:
	data = pk.load(f)


print(len(data))