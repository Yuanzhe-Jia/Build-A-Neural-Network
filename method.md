
# Method

### 1. Multiple Hidden Layers

$$
y_1=w_1^T x+b_1
$$

$$
a_1=σ_1 (y_1)
$$

$$
y_2=w_2^T a_1+b_2
$$

$$
a_2=σ_2 (y_2)
$$

$$
y ̂=w_3^T a_2+b_3
$$

Multiple hidden layers are the basic structure in neural networks. 
The structure helps neural networks satisfy most of data transformation operations and fit various complex problems. 
Taking a simple neural network with two hidden layers as an example, $w_1$ and $b_1$ are the parameters of the first hidden layer, they scale and shift the original input $x$ for many times to obtain $y_1$. 
The number of scaling and shifting depends on the output size (i.e., number of neurons) of this layer. 
For example, if the output size is 10, an input data will be transformed by 10 times at this layer. 
$σ_1$ is the activation function of the first hidden layer, after processing $y_1$ by $σ_1$, $a_1$ is obtained as the output of the first layer. 
Similar to the above, $w_2$ and $b_2$ are the parameters of the second hidden layer, $σ_2$ is the activation function of this layer, and $a_2$ is regarded as the output of the layer. 
$w_3$ and $b_3$ are the parameters of the output layer, they scale and shift the input data $a_2$ to obtain $y ̂$, which is the prediction results of the neural network. 
Basically, the multi-layer architecture of a neural network contains two main characteristics. 
The first is to perform feature extraction. 
The neurons in each layer are equivalent to feature extractors, implementing different transformations on the input data. 
The second is to introduce nonlinear properties through activation functions, thereby improving the generalization ability of neural networks to nonlinear problems. 
As is mentioned above, multiple hidden layers are the basic structure of neural networks, many modules are extended on this basis. Details about those modules will be presented below.


### 2. Kaiming Initialization

$$
w_i=⋃(-√(6⁄n_i ),√(6⁄n_i ))
$$

$$
b_i=0
$$

The initial weights in neural networks should not be too large or too small, larger weights will lead to exploding gradient problem, while smaller weights will also bring vanishing gradient problem. 
The initial weights cannot even be zero, as this would cause all neurons to act exactly the same. 
Kaiming initialization is an effective way to solve these problems. 
In the above formulas, ReLU is considered as the default activation function, $n_i$ is the number of inputs in layer $i$, $w_i$ and $b_i$ represent the weights and biases of layer $i$, respectively. 
By initializing the weights and biases of each layer according to the above formula, the mean of the activations will be zero, and the variance of the activations will be kept the same across layers. 
That is to say, the output of each layer follows the same distribution, which improves the learning efficiency of the neural network. 
If other activation functions are used, the upper and lower bounds of the uniform distribution should also be reset.


### 3. Weight Decay

$$
J ̂(θ_t)=J(θ_t)+λ⁄2 ‖θ_t ‖_2^2
$$

$$
θ_{t+1}=θ_t-η∇_{θ_t} J ̂(θ_t)
$$

Weight decay is one of effective ways to prevent overfitting. 
Weight decay can be regarded as the coefficient $λ$ placed in front of the L2 regularization term, which is a penalty on the sum of squares of parameters. 
The L2 regularization term represents the model complexity, so the role of weight decay is to adjust the impact of model complexity on the loss function. 
If the weight decay is large, the loss of a complex model will also be large. 
As is well-known, the optimization purpose of neural networks is to minimize the loss function $J(θ_t)$. 
If weight decay $λ$ is greater than zero, neural networks will minimize a new loss function $J(θ_t)$, which is equivalent to iterate in the direction where parameters are not too complex, so as to mitigate the overfitting problem. 
In the above formulas, the hyper-parameter $η$ is the learning rate, it is defined as the update step size of the optimizer. 
$θ_t$ represents weights and biases at the time step $t$, it will be updated by the overall update vector $η∇_{θ_t} J ̂(θ_t)$. 
Independent of the above theory, engineers found that an improved model would be obtained by directly decaying the updated parameters at each time step. 
In this document, the engineer's version of weight decay will be adopted.


### 4. Batch Normalization

$$
μ_B←1/m ∑_{i=1}^m x_i
$$

$$
σ_B^2←1/m ∑_{i=1}^m (x_i-μ_B)^2
$$

$$
x ̂_i←(x_i-μ_B)/√(σ_B^2+ε)
$$

$$
y_i←γx ̂_i+β≡BN_{r,β} (x_i)
$$

Batch normalization is a normalization layer commonly used in neural networks, since it can reduce the covariance shift, as well as effects of exploding gradients and vanishing gradients. 
Batch normalization focuses on each feature over all the training data in the mini-batch. 
Specifically, it normalizes input data $x_i$ using the mean $μ_B$ and variance $σ_B^2$ of the mini-batch and introduces two learnable parameters, $γ$ and $β$, which scale and shift the normalised data $x ̂_i$ respectively. 
The hyper-parameter $ε$ is a very small value used to prevent the denominator from being zero. 
Finally, the processed data $y_i$ is the output of the batch normalization layer. 
Batch normalization is sensitive to batch size $m$. When the batch size becomes smaller, the error increases rapidly, which is caused by inaccurate batch statistical estimation. 
In addition, since batch normalization is dependent on the batch size, it cannot be applied at test time the same way as the training time. 
Instead, during test time, batch normalization utilizes moving average and variance to perform inference.


### 5. Dropout





### 6. Label Smoothing




### 7. ReLU Activation Function



### 8. Tanh Activation Function



### 9. GELU Activation Function



### 10. Softmax and Cross-entropy Loss




### 11. Momentum in SGD



### 12. Adam Optimizer




### 13. Mini-batch Training














