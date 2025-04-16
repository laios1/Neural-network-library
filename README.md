# Neural-network-library
a library that I made to generate and train a fully connected neural network like this one.
<p align="center">
  <img src="assets/neural.png" width="400"/>
</p>

# Features
*  Define a custom network architecture with any number of hidden layers (or none).
*  Basic activation functions: Sigmoid, ReLU, and Tanh
*  Adjustable learning rate
*  Fully written in Python and Numpy, no external ML libraries required.

# Requirements
* Python 3.8+
* Numpy

# Installation & Setup

**1. Clone the repository:**

   ```Bash
   git clone https://github.com/laios1/Neural-network-library.git
   cd Neural-network-library
   ```
**2. Create and activate a virtual environment:**

  ### On macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
  ### On Windows:
  ```Bash
  python -m venv venv
  venv\Scripts\activate
  ```

**3. install dependencies**

  ```Bash
  pip install numpy
  ```

you can create it using the class : neural_network
### crée un réseau de neurone : 

`SSize` = [#input_layer, #hidden_layer_1,..., #hidden_layer_n, #output_layer]

/!\ peut ne pas avoir d'hidden layer

`LR`(Learning rate) ; LR = 1 par default 

`act` (activation function) ; act = "sigmoid" par defaut ; (pour l'instant seulement sigmoid, ReLU and tanh)

