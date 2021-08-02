# PyTorch: Training your first Convolutional Neural Network

## Title
[Intro](#Intro)  
[The KMNIST dataset](#The-KMNIST-dataset)
[Project structure](#Project-structure)  
[Implementing a Convolutional Neural Network (CNN) with PyTorch](#Implementing-a-Convolutional-Neural-Network-(CNN)-with-PyTorch)  
[Creating our CNN training script with PyTorch](#Creating-our-CNN-training-script-with-PyTorch)

## Intro
In this tutorial, you will receive a gentle introduction to training your first Convolutional Neural Network(CNN) using the PyTorch deep learning library. This network will be able to recognize handwritten Hiragana characters.
That tutorial focused on simple numerical data. we will take the next step and learn how to train a CNN to recognize handwritten Hiragana characters using the Kuzushiji-MNIST (KMNIST) dataset.

As you’ll see, training a CNN on an image dataset isn’t all that different from training a basic multi-layer perceptron (MLP) on numerical data. We still need to: 

	1. Define our model architecture
	2. Load our dataset from disk
	3. Loop over our epochs and batches
	4. Make predictions and compute our loss
	5. Properly zero our gradient, perform backpropagation, and update our model parameters

Furthermore, this post will also give you some experience with PyTorch’s `DataLoader` implementation which makes it super easy to work with datasets --- becoming proficient with PyTorch’s `DataLoader` is a critical skill you’ll want to develop as a deep learning practitioner. 

To learn how to train your first CNN with PyTorch, just keep reading. Throughout the reminder of this tutorial, you will learn how to train your first CNN using the PyTorch framework.
We’ll start by configuring our development environment to install both torch and torchvision, followed by reviewing our project directory structure.
I’ll then show you the KMNIST dataset (a drop-in replacement for the MNIST digits dataset) that contains Hiragana characters. Later in this tutorial, you’ll learn how to train a CNN to recognize each of the Hiragana characters in the KMNIST dataset.
We’ll then implement three Python scripts with PyTorch, including our CNN architecture, training script, and a final script used to make predictions on input images. 
Configuring your development environment to follow this guide, you need to have PyTorch, OpenCV, and scikit-learn installed on your system.Luckily, all three are extremely easy to install pip:

PyTorch: Training your first Convolutional Neural Network (CNN)
```bash
$ pip install torch torchvision
$ pip install opencv-contrib-python
$ pip install scikit-learn
```
**If you need help configuring your development environment for PyTorch, I highly recommend that you read the PyTorch documentation** --- PyTorch’s documentation is comprehensive and will have you up and running quickly.

## The KMNIST dataset
<figure>
<img src="./images/Figure1.png" width=100% align="center">
	<span style="font-size: 0.8em; color:gray;"><figcaption align="center">
		Figure 1: The KMNIST dataset is a drop-in replacement for the standard MNIST dataset. The KMNIST dataset contains examples of handwritten Hiragana characters.
	</figcaption></span>
</figure>



The dataset we are using today is the Kuzushiji-MNIST dataset, or KMNIST, for short. This dataset is meant to be a drop-in replacement for the standard MNIST digits recognition dataset.

The KMNIST dataset consists of 70,000 images and their corresponding labels (60,000 for training and 10,000 for testing).

There are a total of 10 classes (meaning 10 Hiragana characters) in the KMNIST dataset, each equally distributed and represented. Our goal is to train a CNN that can accurately classify each of these 10 characters.

And lucky for us, the KMNIST dataset is built into PyTorch, making it super easy for us to work with!

## Project structure
Before we start implementing and PyTorch code, let’s first review our project directory structure.

You’ll then be presented with the following directory structure:
```bash
$ tree . --dirsfirst
.
├-- output
│       ├-- model.pth
│       └-- plot.png
├-- moduels
│       ├-- __init__.py
│       └-- lenet.py
├-- predict.py
└---- train.py
2 directories, 6 files
```

We have three Python scripts to review today:  

> 1. `lenet.py`: Our PyTorch implementation of the famous LeNet architecture  
> 2. `train.py`: Trains LeNet on the KMNIST dataset using PyTorch, then serializes the trained model to disk(i.e., model.pth)  
> 3. `predict.py`: Loads our trained model from disk, makes predictions on testing images, and displays the results on our screen  

The output directory will be populated with plot.png (a plot of our training/validation loss and accuracy) and model.pth (our trained model file) once we run train.py

With our project directory structure reviewed, we can move on to implementing our CNN with PyTorch.

## Implementing a Convolutional Neural Network (CNN) with PyTorch
<figure>
<img src="./images/Figure2.png" width=100% align="center">
	<span style="font-size: 0.8em; color:gray;"><figcaption align="center">
		Figure 2: The LeNet architecture. We’ll be implementing LeNet with PyTorch
	</figcaption></span>
</figure>

The Convolutional Neural Network(CNN) we are implementing here with PyTorch is the seminal LeNet architecture, first proposed by one of the grandfathers of deep learning, Yann LeCunn.

By today’s standards, LeNet is a *very shallow* neural network, consisting of the following layers:

	(CONV => RELU => POOL) * 2 => FC => RELU => FC => SOFTMAX

As you’ll see, we’ll be able to implement LeNet with PyTorch in only 60 lines of code (including comments).

The best way to learn about CNNs with PyTorch is to implement one, so with that said, open the `lenet.py` file in the `modules` module, and let’s get to work:

PyTorch: Training your first Convolutional Neural Network (CNN)

```python
# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
```
import our required packages. Let’s break each of them down:

> * `Module`: Rather than using the `Sequential` PyTorch class to implement LeNet, we'll instead subclass the Module object so you can see how PyTorch implements neural networks using classes
> * `Conv2d` : PyTorch's implementation of convolutional layers
> * `Linear` : Fully connected layers
> * `MaxPool2d` : Applies 2D max-pooling to reduce the spatital dimensions of the input volume
> * `ReLU`: Our ReLU activation function
> * `LogSoftmax` : Used when building our softmax classifier to return the predicted probabilities of each class
>* `flatten` : Flattens the output of a multi-dimensional volume (e.g., a CONV or POOL layer) such that we can apply fully connected layers to it

With our imports taken care of, we can implement our `LeNet` class using PyTorch:

```python
class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()

		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()

		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)
```
**Line 1** defines the `Lenet` class. Notice how we are subclassing the `Module` object object -- by building our model as a class we can easily

* Reuse variables
* Implement custom functions to generate subnetworks/components(used *very often* when implementing more complex networks, such as ResNet, Inception, etc.)
* Define our own `forward` pass function

**best of all, when defined correctly, PyTorch can automatically apply its autograd module to perform automatic differentiation -- backpropagation is taken care of for us by virtue of the PyTorch library!**  

The constructor to `LeNet` accepts two variables:

> 1. `numChannels` : The number of channels in the input images (`1` for grayscale or `3` for RGB)
> 2. `classes` : Total number of unique class labels in our dataset 

* calls the parent constructor (i.e., `Module`) which performs a number of PyTorch-specific operations.

* calls the parent constructor (i.e., `Module`) which performs a number of PyTorch-specific operations.

* Initialize our first set of `CONV => RELU => POOL` layers. Our first CONV layer learns a total of 20 filters, each of which are *5 $\times$ 5*. A ReLU activation function is then applied, followed by a *2 \times 2* max-pooling layer with a *2 $\times$ 2*stride to reduce the spatital dimensions of our input image.

We then have a second set of `CONV => RELU => POOL` layers. We increase the number of filters learned in the CONV layer to 50, but maintain the *5 $\times$ 5* kernel size. Again, a ReLU activation is applied, followed by max-pooling.

Next comes our first and only set of fully connected layers. We define the number of inputs to the layer (`800`) along with our desired number of output nodes (`500`). A ReLU activation follows the FC layer.

Finally, we apply our softmax classifier. The number of `in_features` is set to `500`, which is the *output* dimensionality from the previous layer. We then apply `LogSoftmax` such that we can obtain predicted probabilities during evaluation.

**To build the network architecture itself (i.e, what layer is input to some other layer), we need to override the `forward` method method of the `Module` class**.

The `forward` function serves a number of purpose:  
1. It connects layers/subnetworks together from variables defined in the constructor (i.e, \_\_init__) of the class
2. It defines the network architecture itself
3. It allows the forward pass of the model to be performed, resulting in our output predictions
4. And, thanks to PyTorch's autograd module, it allows us to perform automatic differentiation and update our model weights.

```python
	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)

		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)

		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)

		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)

		# return the output predictions
		return output
```

The `forward` method accepts a single parameter, `x`, which is the batch of input data to the network.

We then connect our `conv1`, `relu1`, and `maxpool1` layers together to form the first `CONV => RELU => POOL` layer of the network.

A similar operation is performed, this time building the second set of `CONV => RELU => POOL` layers.  

At this point, the variable `x` is a multi-dimensional tensor; however, in order to create our fully connected layers, we need to "flatten" this tensor into what essentially amounts to a 1D list of values --- the `flatten` function takes care of this operation for us.

From there, we connect the `fc1` and `relu3` layers to the network architecture, followed by attaching the final `fc2` and `logSoftmax`.

The `output` of the network is then returned to the calling function.  

**Again, I want to reiterate the importance of *initializing variables in the constructor* versus *building the netowrk itself in the `forward` function:***

* The constructor to your `Module` only initializes your layers types. PyTorch keeps track of these variables, but it has no idea how the layers connect to each other.

* For PyTorch to understand the network architecture you're building, you define the `forward` function.  

* Inside the `forward` function you take the variables initialized in your constructor and connect them.

* PyTorch can then make predictions using your network and perform automatic backpropagation, thanks to the autograd module

## Creating our CNN training script with PyTorch

With our CNN architecture implemented, we can move on to creating our training script with PyTorch.

Open the `train.py` file in your project directory structure, and let's get to work:

```python
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
```

Import `matplotlib` and set the appropriate background engine.

From there, we import a number of notable packages:  

* `Lenet`: Our PyTorch implementation of the LeNet CNN from the previous section  
* `classification_report`: Used to display a detailed classification report on our testing set
* `random_split`: Constructs a random training/testing split from an input set of data
* `DataLoader`: PyTorch's *awesome* data loading utility that allows us to effortlessly build data pipelines to train our CNN