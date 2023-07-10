class AliasDict(dict):
	def __init__(self, *args, **kwargs):
		dict.__init__(self, *args, **kwargs)
		self.aliases = {}

	def __getitem__(self, __key):
		return dict.__getitem__(self, self.aliases.get(__key, __key))

	def __setitem__(self, __key, __value) -> None:
		return dict.__setitem__(self, self.aliases.get(__key, __key), __value)

	def add(self, __key, *aliases):
		for alias in aliases:
			self.aliases[alias] = __key


def progress_bar(metrics):
	...


def rename_key(dictionary, k_old, k_new):
	pos = list(dictionary.keys()).index(k_old)
	value = (k_new, dictionary[k_old])
	dictionary.pop(k_old)

	items = list(dictionary.items())
	items.insert(pos, value)

	return dict(items)

