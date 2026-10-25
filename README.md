# 🔢 Digit-Recognizor

> A handwritten digit classifier built **from scratch using pure mathematics**: no TensorFlow, no PyTorch, just linear algebra, calculus, and NumPy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project implements a fully connected neural network that recognizes handwritten digits (0 to 9) from the [MNIST]([http://yann.lecun.com/exdb/mnist/](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/code)) dataset. Every part of the learning process (forward propagation, loss, backpropagation, and gradient descent) is written by hand from the underlying equations, with no deep learning framework doing the work.

**Goal:** understand *why* neural networks work, not just *how to call them*.

---

## ✨ Features

- Neural network implemented with only NumPy matrix operations
- Hand-derived forward and backward propagation
- Gradient descent weight updates
- Trained and evaluated on MNIST (28×28 grayscale digit images)
- Everything in one readable Jupyter notebook

---

## 🧠 The Mathematics

Each image is flattened into a vector of 784 pixel values (28 × 28) and normalized to the range [0, 1].

### 1. Forward Propagation

For each layer $l$:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

$$A^{[l]} = g(Z^{[l]})$$

Hidden layers use a non-linear activation such as ReLU:

$$\text{ReLU}(z) = \max(0, z)$$

The output layer uses Softmax to turn raw scores into probabilities over the 10 digit classes:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

### 2. Loss Function

Cross-entropy loss measures how far the predicted probabilities are from the true label (one-hot encoded):

$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10} y_k^{(i)} \log\left(\hat{y}_k^{(i)}\right)$$

### 3. Backpropagation

Using the chain rule, gradients flow backward through the network. For softmax with cross-entropy, the output error simplifies to:

$$dZ^{[L]} = A^{[L]} - Y$$

For earlier layers:

$$dW^{[l]} = \frac{1}{m}\, dZ^{[l]} \left(A^{[l-1]}\right)^T$$

$$db^{[l]} = \frac{1}{m}\sum dZ^{[l]}$$

$$dZ^{[l-1]} = \left(W^{[l]}\right)^T dZ^{[l]} \odot g'\left(Z^{[l-1]}\right)$$

### 4. Gradient Descent

Parameters are updated in the direction that reduces the loss, with learning rate $\alpha$:

$$W^{[l]} := W^{[l]} - \alpha\, dW^{[l]}$$

$$b^{[l]} := b^{[l]} - \alpha\, db^{[l]}$$

---

## 🏗️ Network Architecture

```
Input (784)  →  Hidden Layer(s) (ReLU)  →  Output (10, Softmax)
 28×28 pixels                               digits 0–9
```

<!-- TODO: replace with your exact layer sizes, e.g. 784 → 128 → 10 -->

| Component         | Choice                  |
|-------------------|-------------------------|
| Input size        | 784                     |
| Hidden layers     | _fill in your sizes_    |
| Hidden activation | ReLU                    |
| Output activation | Softmax                 |
| Loss              | Cross-entropy           |
| Optimizer         | Gradient descent        |
| Learning rate (α) | _fill in_               |
| Epochs / iterations | _fill in_             |

---

## 📁 Repository Structure

```
Digit-Recognizor/
├── MNIST.ipynb        # Full implementation: data, model, training, evaluation
├── mnist_test.csv     # MNIST test set (label + 784 pixel columns per row)
├── LICENSE            # MIT License
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- NumPy, Pandas, Matplotlib

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ShivanshMishra659/Digit-Recognizor.git
cd Digit-Recognizor

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas matplotlib jupyter
```

### Run the notebook

```bash
jupyter notebook MNIST.ipynb
```

Then run all cells from top to bottom (**Kernel → Restart & Run All**).

> **Note:** The data format is a CSV where the first column is the digit label (0–9) and the remaining 784 columns are pixel intensities (0–255). If the notebook trains on a separate `mnist_train.csv` that is not included in this repo because of its size, download it from [Kaggle's MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and place it in the project root.

---

## 📊 Results

<!-- TODO: add your numbers and a screenshot or two from the notebook -->

| Metric            | Value        |
|-------------------|--------------|
| Training accuracy | _fill in_ %  |
| Test accuracy     | _fill in_ %  |

You can add sample predictions here:

```
![Sample predictions](images/predictions.png)
```

---

## 🔍 How It Works (Summary)

1. **Load** the CSV and split labels from pixels
2. **Normalize** pixels by dividing by 255
3. **Initialize** weights randomly and biases to zero
4. **Forward pass**: compute predictions
5. **Compute loss**: compare to true labels
6. **Backward pass**: compute gradients with the chain rule
7. **Update** weights using gradient descent
8. **Repeat** for many iterations, then evaluate on unseen test data

---

## 🛣️ Future Improvements

- [ ] Mini-batch gradient descent and momentum / Adam optimizer
- [ ] Add more hidden layers and compare accuracy
- [ ] L2 regularization and dropout to reduce overfitting
- [ ] Confusion matrix and per-digit error analysis
- [ ] Interactive demo where you draw a digit and the model predicts it (Streamlit or Gradio)
- [ ] Compare against an equivalent PyTorch / Keras model

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome. Feel free to open an issue or submit a pull request.

1. Fork the repo
2. Create your branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Shivansh Mishra**
GitHub: [@ShivanshMishra659](https://github.com/ShivanshMishra659)

⭐ If you found this project useful, consider giving it a star!
