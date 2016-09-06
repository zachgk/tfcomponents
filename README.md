# tfcomponents

TFComponents is a component structure for building complex structures in tensorflow.  It currently is designed to work with tflearn.

The list of predefined components are as follow:
- Identity()
- Sequence(listOfComponentBlocks)
- Parallel(listOfComponentBlocks)
- Chain(size, componentBlock)
- Fractal(size, componentBlock)
- Residual(componentBlock)

These components also have several keyword arguments.  They can take the property name for the name of the enclosing scope.  It can also use a globalDroppath boolean, localDroppath boolean, and localDroppathProb tensor float between 0 and 1.  The booleans can either be a python boolean or a potentially changing 0D boolean tensor.

In addition, the tflearn layers Conv2d and ShallowResidualBlock have been converted to component form (they no longer possess the number of blocks property)


## Usage

All components possess a getitem method, like this:
```python
outputTensor = RandomComponent(...some arguments...)[inputTensor]
```

These components can be combined by passing some component definitions into others.

```python
# Basic residual net
net = Chain(10, Residual(ShallowResidualBlock(16)))[net]

globalDroppath = tf.less(tf.random_uniform([]), 0.5)
localDroppath = tf.logical_not(globalDroppath)
localDroppathProb = 0.85 #keeps this fraction

#Fractalnet with droppath
net = Fractal(2, FractalBlock(filters), globalDroppath=globalDroppath, localDroppath=localDroppath, localDroppathProb=localDroppathProb)[net]

#Residual net with droppath
net = Chain(10, Residual(ShallowResidualBlock(16)), globalDroppath=globalDroppath, localDroppath=localDroppath, localDroppathProb=localDroppathProb)[net]
```


## Custom Components
To create a custom component, extend the TFComponent class.
It must have an init method that sets self.opts to a dictionary (with options, ideally).  The init method can also take ordered arguments which can just be saved directly to the object.

It should also define a method get(self, incoming, opts, inherit).
Incoming is an incoming tensor, opts is the self.opts dictionary converted to a 

