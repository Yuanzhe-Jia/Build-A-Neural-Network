# Build a neural network from scratch

## Introduction

- What is a neural network?
- What modules are included in a neural network?
- What can neural networks be used for?

These questions may have plagued those interested in neural networks. In this study, the author builds a neural network structure from scratch, and applies it to accomplish a multi-class classification task on a given dataset. The neural network modules achieved in the study include: 
1. multiple hidden layers
2. Kaiming initialization
3. weight decay
4. batch normalization
5. dropout
6. label smoothing
7. ReLU activation function
8. Tanh activation function
9. GELU activation function
10. softmax and cross-entropy loss
11. momentum in SGD
12. Adam optimizer
13. mini-batch training

Additionally, various experiments have been conducted based on the self-built neural network structure. The prediction loss, prediction accuracy and running time are adopted as the evaluation methods of the experimental models. Hyper-parameter tuning, comparative analysis and ablation studies are also included.

## How to use

Step 1. Save the `model.py` file in the same root with your code files.

Step 2. Import functions from the `model.py` file.
```
from model import dense, batch_norm, dropout, model #neural network architecture
from model import relu, gelu, tanh #activation functions
from model import softmax, softmax_cross_entropy_loss, softmax_cross_entropy_derivatives #loss functions
from model import categorical_accuracy #evaluation metrics
```

Step 3. See examples in the `experiements.ipynb` file to learn how to build a neural network usding these functions.
