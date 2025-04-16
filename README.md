# Neural-network-library
a library that I made to generate and train a fully connected neural network like this one.
<p align="center">
  <img src="assets/neural.png" width="400"/>
</p>

# Features
*  Define a custom network architecture with any number of hidden layers (or none).
*  Basic activation functions: Sigmoid, ReLU, and Tanh
* Adjustable learning rate
* Fully written in Python and Numpy, no external ML libraries required.

you can create it using the class : neural_network
### crée un réseau de neurone : 

`SSize` = [#input_layer, #hidden_layer_1,..., #hidden_layer_n, #output_layer]

/!\ peut ne pas avoir d'hidden layer

`LR`(Learning rate) ; LR = 1 par default 

`act` (activation function) ; act = "sigmoid" par defaut ; (pour l'instant seulement sigmoid, ReLU and tanh)

