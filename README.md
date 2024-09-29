# Neural Networks with Keras

## Project Overview
This project explores the fundamentals of Neural Networks using Keras, a powerful library for building deep learning models in Python. The notebook provides theoretical insights, practical implementations, and visualizations to enhance understanding of how neural networks operate, including various activation functions and the Perceptron model.

## What I've Learned

- **Neural Network Basics**: Gained insights into the structure and functioning of neural networks, including layers, weights, and activation functions.
- **Activation Functions**: Explored various activation functions like Sigmoid, ReLU, and Tanh, along with their derivatives.
- **Perceptron Model**: Implemented the Perceptron algorithm using the Iris dataset to classify different species of Iris flowers.

## Getting Started

### Prerequisites

To run this notebook, ensure you have the following:

- Python version ≥ 3.5
- TensorFlow version ≥ 2.0
- Required libraries:
  - NumPy
  - Matplotlib
  - Scikit-Learn

You can install the necessary libraries using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

### Running the Notebook

To execute the notebook, follow these steps:

1. Clone or download the repository containing the notebook.
2. Open the notebook in a Jupyter environment.
3. Run the cells sequentially to load data, build neural networks, and visualize results.

## Key Concepts Covered

### 1. Setup

The notebook begins by importing necessary libraries and setting up the environment for reproducibility. Key imports include:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```

### 2. Data Loading and Preprocessing

The Iris dataset is loaded to demonstrate classification using the Perceptron model:

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length and petal width
y = (iris.target == 0).astype(np.int)  # Iris-setosa
```

### 3. Perceptron Implementation

A Perceptron classifier is created and fitted to the data. The following code demonstrates its usage:

```python
from sklearn.linear_model import Perceptron

per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(X, y)
```

### 4. Decision Boundary Visualization

The decision boundary created by the Perceptron is visualized using the following code:

```python
a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
b = -per_clf.intercept_ / per_clf.coef_[0][1]

# Create meshgrid for visualization
x0, x1 = np.meshgrid(
    np.linspace(0, 5, 500).reshape(-1, 1),
    np.linspace(0, 2, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.contourf(x0, x1, zz, cmap=custom_cmap)
```

### 5. Activation Functions

Several activation functions are defined, including their derivatives:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)
```

### 6. Visualization of Activation Functions

The different activation functions and their derivatives are visualized using:

```python
z = np.linspace(-5, 5, 200)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(z, sigmoid(z), "g--", label="Sigmoid")
plt.plot(z, relu(z), "m-.", label="ReLU")
```

## Conclusion

This project serves as an educational resource for anyone interested in machine learning and deep learning, specifically focusing on Neural Networks. By exploring the Perceptron model and activation functions, users gain a deeper understanding of how neural networks learn and make predictions.
