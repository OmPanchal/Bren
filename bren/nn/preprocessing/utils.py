import random
import bren as br
import numpy as np

def shuffle(*arrays, seed_domain=1000):
    seed = random.randint(0, seed_domain)

    for array in arrays:
        np.random.seed(seed)
        np.random.shuffle(array)

    
def split_uneven(array, split):
    remainder = len(array) % split
    return br.Variable(np.split(array[:len(array) - remainder], split))