  <div id="header" align="center">
    <img id="logo" src="https://github.com/OmPanchal/Bren/blob/main/bren/B.png" ></img>   
  </div>

[bren](https://pypi.org/project/bren/) is a custom [numpy](https://numpy.org) based library, powered by automatic differentiation, inspired by [Tensorflow](https://www.tensorflow.org)/[Keras](https://keras.io), which allows users to build small scale simple neural networks. It's analogous yet simpler design to the Keras API allows users to produce, train and save their own models, with custom components, without having to learn an entirely new structure. 

bren is part of a sequence of neural network from scratch projects and a successor to the [neural-network-from-scratch-v2](https://github.com/OmPanchal/nn-from-scratch-2), with one major update being the integration of automatic differentitation. Automatic differentiation allows for the real-time determination of derivatives during backpropagation (through the use of computation graphs produced by `br.autodiff` and `br.Variable`) and reduces the need for users to couple mathematical computation with pre-written derivatives as was required in the previous projects.

## Install
To install the latest version of bren, run: 
```
pip install bren
```

**Your first bren program**. (Examples tend to import `bren` as `br`)
```python
import bren as br

A = br.Variable([1, 2, 3])
print(A + 2) # <Variable value=[3. 4. 5.] dtype=float64>
```

## `br.autodiff` (Automatic Differentiation)
bren is an automatic differentiation driven neural network library, with backpropagation making use of `br.Graph` to find the derivatives of the trainable parameters with respect to the loss. This is governed by `br.Variable` which keeps track of any operation which have been performed on the `Variable` object. `br.autodiff` is used to produce a computation graph of these recorded computations, this graph can be back tracked to determine the derivatives of a given *dy* value with respect to a given *dx* value. As a gradient, a `br.Constant` is returned (the derivative of a `br.Constant` is always 0). The functionings of `br.Variable` and `br.Constant` are a result of the swift array computation of [numpy](https://numpy.org).

```python
import bren as br

A = br.Variable([1, 2, 3], dtype="float64")

with br.Graph() as g:
	B = A ** 2 # any computation performed on a Variable will be tracked 
	print(g.grad(B, [A])) # [<Constant value=[2. 4. 6.] dtype=float64>]
```

## `br.nn` (Neural Networks)
bren's modular design is heavily inspired by the [Keras](https://keras.io) API, allowing users to produce networks comprised of a variety of customisable components. Users are also capable of producing their own custom components such as (layers, activations, initialisers, losses, metrics) through the use of the respective base classes. `br.nn` allows ready made components to be imported and custom components to be produced.

**Your first neural network with bren**
