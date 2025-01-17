# Build A Neural Network From Scratch

### Introduction

- What is a neural network?
- What modules are included in a neural network?
- What can neural networks be used for?

These questions may have plagued those interested in neural networks. 
In this study, the author builds a neural network structure from scratch, and applies it to accomplish a multi-class classification task. 
The [neural network modules](https://github.com/Yuanzhe-Jia/Build-A-Neural-Network/blob/main/method.md) include: 

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


### How to use

Step 1. 
Install [packages](https://pypi.org/project/network-yuanzhe/).
```
!pip install network_yuanzhe==1.0.2
```

Step 2. 
Import functions from the package.
```
from network_yuanzhe import model
```

Step 3. 
Download the `/test/exp.html` file and open it in your browser.
The experiment show you how to build a classifier based on the self-built neural network structure. 
The prediction loss, prediction accuracy and running time are adopted as the evaluation methods. 
Hyper-parameter tuning, comparative analysis and ablation studies are also included.
