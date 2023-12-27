# Build a neural network from scratch

## Introduction

- What is a neural network?
- What modules are included in a neural network?
- What can neural networks be used for?

These questions may have plagued those interested in neural networks. In this study, the author builds a neural network structure from scratch, and applies it to accomplish a multi-class classification task on a given dataset. The neural network modules achieved in the study include: 
1. Multiple Hidden Layers
2. Kaiming Initialization
3. Weight Decay
4. Batch Normalization
5. Dropout
6. Label Smoothing
7. ReLU Activation Function
8. Tanh Activation Function
9. GELU Activation Function
10. Softmax and Cross-entropy Loss
11. Momentum in SGD
12. Adam Optimizer
13. Mini-batch Training

## How to use

Step 1. Save the `model.py` file in the same root with your code files.

Step 2. Import functions from the `model.py` file.
```
from model import dense, batch_norm, dropout, model #neural network architecture
from model import relu, gelu, tanh #activation functions
from model import softmax, softmax_cross_entropy_loss, softmax_cross_entropy_derivatives #loss functions
from model import categorical_accuracy #evaluation metrics
```

Step 3. See an image classification example in the `experiements.ipynb` file to learn how to build a classifier based on the self-built neural network structure. The prediction loss, prediction accuracy and running time are adopted as the evaluation methods. Hyper-parameter tuning, comparative analysis and ablation studies are also included.
