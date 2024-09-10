
# Method

### 1. Multiple Hidden Layers

$$
y_1 = w_1^T x + b_1
$$

$$
a_1 = σ_1 (y_1)
$$

$$
y_2 = w_2^T a_1 + b_2
$$

$$
a_2 = σ_2 (y_2)
$$

$$
\hat{y} = w_3^T a_2+b_3
$$

Multiple hidden layers are the basic structure in neural networks. 
The structure helps neural networks satisfy most of data transformation operations and fit various complex problems. 
Taking a simple neural network with two hidden layers as an example, $w_1$ and $b_1$ are the parameters of the first hidden layer, they scale and shift the original input $x$ for many times to obtain $y_1$. 
The number of scaling and shifting depends on the output size (i.e., number of neurons) of this layer. 
For example, if the output size is 10, an input data will be transformed by 10 times at this layer. 
$σ_1$ is the activation function of the first hidden layer, after processing $y_1$ by $σ_1$, $a_1$ is obtained as the output of the first layer. 
Similar to the above, $w_2$ and $b_2$ are the parameters of the second hidden layer, $σ_2$ is the activation function of this layer, and $a_2$ is regarded as the output of the layer. 
$w_3$ and $b_3$ are the parameters of the output layer, they scale and shift the input data $a_2$ to obtain $\hat{y}$, which is the prediction results of the neural network. 
Basically, the multi-layer architecture of a neural network contains two main characteristics. 
The first is to perform feature extraction. 
The neurons in each layer are equivalent to feature extractors, implementing different transformations on the input data. 
The second is to introduce nonlinear properties through activation functions, thereby improving the generalization ability of neural networks to nonlinear problems. 
As is mentioned above, multiple hidden layers are the basic structure of neural networks, many modules are extended on this basis. Details about those modules will be presented below.


### 2. Kaiming Initialization

$$
w_i \sim ⋃(-\sqrt{\frac{6}{n_i}}, \sqrt{\frac{6}{n_i}})
$$

$$
b_i = 0
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
\hat{J}(θ_t) = J(θ_t) + \frac{λ}{2} ‖θ_t‖_2^2
$$

$$
θ_{t+1} = θ_t - η∇_{θ_t} \hat{J}(θ_t)
$$

Weight decay is one of effective ways to prevent overfitting. 
Weight decay can be regarded as the coefficient $λ$ placed in front of the L2 regularization term, which is a penalty on the sum of squares of parameters. 
The L2 regularization term represents the model complexity, so the role of weight decay is to adjust the impact of model complexity on the loss function. 
If the weight decay is large, the loss of a complex model will also be large. 
As is well-known, the optimization purpose of neural networks is to minimize the loss function $J(θ_t)$. 
If weight decay $λ$ is greater than zero, neural networks will minimize a new loss function $J(θ_t)$, which is equivalent to iterate in the direction where parameters are not too complex, so as to mitigate the overfitting problem. 
In the above formulas, the hyper-parameter $η$ is the learning rate, it is defined as the update step size of the optimizer. 
$θ_t$ represents weights and biases at the time step $t$, it will be updated by the overall update vector $η∇_{θ_t} \hat{J}(θ_t)$. 
Independent of the above theory, engineers found that an improved model would be obtained by directly decaying the updated parameters at each time step. 
In this document, the engineer's version of weight decay will be adopted.


### 4. Batch Normalization

$$
μ_B ← \frac{1}{m} ∑_{i=1}^m x_i
$$

$$
σ_B^2 ← \frac{1}{m} ∑_{i=1}^m (x_i-μ_B)^2
$$

$$
\hat{x_i} ← \frac{x_i-μ_B}{\sqrt{σ_B^2+ε}}
$$

$$
y_i ← γ\hat{x_i} + β ≡ BN_{r,β} (x_i)
$$

Batch normalization is a normalization layer commonly used in neural networks, since it can reduce the covariance shift, as well as effects of exploding gradients and vanishing gradients. 
Batch normalization focuses on each feature over all the training data in the mini-batch. 
Specifically, it normalizes input data $x_i$ using the mean $μ_B$ and variance $σ_B^2$ of the mini-batch and introduces two learnable parameters, $γ$ and $β$, which scale and shift the normalised data $\hat{x_i}$ respectively. 
The hyper-parameter $ε$ is a very small value used to prevent the denominator from being zero. 
Finally, the processed data $y_i$ is the output of the batch normalization layer. 
Batch normalization is sensitive to batch size $m$. When the batch size becomes smaller, the error increases rapidly, which is caused by inaccurate batch statistical estimation. 
In addition, since batch normalization is dependent on the batch size, it cannot be applied at test time the same way as the training time. 
Instead, during test time, batch normalization utilizes moving average and variance to perform inference.


### 5. Dropout

$$
r_j^{(l)} \sim Bernoulli(p)
$$

$$
\hat{x}^{(l)} = r^{(l)} * x^{(l)}
$$

$$
y_i^{(l+1)} = w_i^{(l+1)} \hat{x}^{(l)} + b_i^{(l+1)}
$$

$$
x_i^{(l+1)} = f(y_i^{(l+1)})
$$

Dropout is a simple but effective technique for mitigating the overfitting problem.
According to this method, when training neural networks, each neuron is retained with a certain probability of $p$, which is usually set to 0.5. 
And $r_j^{(l)}$ is the probability rate for the neuron $j$ in the layer $l$, it will determine if the neuron should use the input $x^{(l)}$.
While in test phase, all neurons are involved in prediction. The intuition of this approach is similar to ensemble learning, which is a technique intended to deal with overfitting problem by combining predictions from many different models. 
However in real scenarios, the inverted dropout is usually applied. 
To be Specific, during training time, parameters in the neural network are amplified by a factor of $1/p$. While in test time, the network is used as a whole and no parameter scaling will be performed.


### 6. Label Smoothing

$$
y_k^{Ls} = y_{k} (1-α) + \frac{α}{K}
$$

Label smoothing is a technique that can prevent the overfitting problem and improve the generalization ability of the model. 
To be specific, label smoothing is to add noise on the basis of one-hot encoding. 
This method is simple but effective, and has achieved remarkable results in many image classification tasks. 
It is obvious that one-hot encoding will make the prediction probability of the correct classification get closer and closer to 1. 
In other words, one-hot encoding is not soft enough, resulting in the model being too confident in its prediction. 
By adding noise to the one-hot labels, the label smoothing technique can alleviate the problem of the model being too arbitrary, thereby enhancing the generalization ability of the model. 
In the above formula, $y_k$ is the one-hot label for a particular category $k$, and $α$ is a hyper-parameter, which is usually set to 0.1. 
$K$ is the number of categories. 
$y_k^{Ls}$ is the label of the category $k$ processed by the label smoothing technique.


### 7. ReLU Activation Function

$$
relu(x) = max⁡(0,x)
$$

The full name of ReLU is Rectified Linear Unit. 
It performs a threshold operation on each input element $x$, where values less than zero are set to zero. 
ReLU activation function is by far the most widely used activation function in deep learning applications, since it has two main advantages. 
The one is the fast calculation, because it does not involve exponentiation and division, which improves the overall calculation speed. 
The second is that ReLU activation function introduces sparsity in the hidden units as it compresses some input values to zero.


### 8. Tanh Activation Function

$$
tanh(x) = \frac{ⅇ^{x}-ⅇ^{-x}}{ⅇ^{x}+ⅇ^{-x}}
$$

The full name of Tanh is Hyperbolic Tangent Function. 
It squeezes each input element $x$ between -1 and 1. 
Tanh activation function has been widely adopted in the recurrent neural networks (RNN) for natural language processing (NLP) tasks. 
The benefit of this function is to speed up the back-propagation process due to its zero-centred nature. 
However, this function cannot effectively solve the vanishing gradient problem. When the input elements are too large or too small, the gradients of the function will be close to zero.


### 9. GELU Activation Function

$$
gelu(x) = 0.5x(1 + tanh⁡(\sqrt{\frac{2}{π}} (x + 0.44715x^3)))
$$

The full name of GELU is Gaussian Error Linear Unit. 
Similar to ReLU activation function, GELU activation function can also preserve the input element $x$, or compress it to zero. 
But their difference is that in GELU activation function, 
keeping the original value or compressing it to zero depends on probability that the current input is greater than the rest of the inputs. 
It can be seen that when the input element is larger, it is more likely to be retained, 
and the smaller the input is, the more likely it is to be reset to zero. 
In recent years, GELU activation function has been widely used in Transformer models including BERT, GPT2, etc.


### 10. Softmax and Cross-entropy Loss

$$
a_i = softmax(z_i) = \frac{ⅇ^{z_i}}{∑_{j=1}^K ⅇ^{z_j}}
$$

$$
J = -∑_{i=1}^K y_i log⁡(a_i)
$$

The output values of a neural network are difficult to interpret, 
so they need to be converted into a probability distribution between 0 and 1 with the help of the softmax function. 
In the above formulas, $K$ represents the number of categories, 
$z_i$ represents the input value of the current category $i$, which will be normalised by the softmax function. 
The output value $a_i$ is between 0 and 1, and the sum of softmax outputs is equal to 1. 
Cross-entropy is used to evaluate the loss $J$ between the probability distribution obtained by the neural network and the true distribution. 
It depicts the distance between the actual output probability and the expected output probability, 
that is, the smaller the cross-entropy, the closer the two probability distributions are. 
And $y_i$ represents the expected output probability of the category $i$, which usually needs to be converted into one-hot format.


### 11. Momentum in SGD

$$
v_t = γv_{t-1} + η∇_{θ_t} J{θ_t}
$$

$$
θ_{t+1} = θ_t - v_t
$$


$$
θ_{t+1} = θ_t - η∇_{θ_t} J{θ_t}
$$

Momentum is a commonly used acceleration method for (Stochastic Gradient Descent) SGD optimizer. 
It adds a portion of the update vector from past time steps to the current update vector, resulting in the effect of speeding up optimization and suppressing oscillations. 
Specifically, if the current gradient $∇_{θ_t} J(θ_t)$ and the momentum term $v_{t-1}$ point in the same direction, the overall update vector $v_t$ will increase, which is equivalent to speeding up the optimizer. 
If the current gradient and the momentum term point in the opposite direction, then the overall update vector is reduced, which is equivalent to reducing oscillations. 
In the above formulas, the hyper-parameter $γ$ is a momentum factor, which is usually set to 0.9. 
The hyper-parameter $η$ is the learning rate, it is defined as the update step size of the optimizer. 
And $θ_t$ represents weights and biases at time step $t$, it will be updated by the overall update vector $v_t$.


### 12. Adam Optimizer

$$
g_t = ∇_{θ_t} J(θ_t)
$$

$$
m_t = β_1 m_{t-1} + (1-β_1) g_t 
$$

$$
ν_t = β_2 v_{t-1} + (1-β_2) g_t^2  
$$

$$
\hat{m_t} = \frac{m_t}{1-β_1^t} 
$$

$$
\hat{v_t} = \frac{v_t}{1-β_2^t}
$$

$$
θ_{t+1} = θ_t - \frac{η}{\sqrt{\hat{v_t}+ε}} \hat{m_t}
$$

The full name of Adam is Adaptive Moment Estimation. 
It computes an adaptive learning rate for each parameter in the neural network. 
Similar to momentum in SGD mentioned above, Adam optimizer maintains an exponentially decaying average of past gradients $m_t$. 
In addition, Adam optimizer also applies the exponentially decaying average of past squared gradients $ν_t$. 
$m_t$ and $ν_t$ are estimates of the first and second moments of the gradients $g_t$, respectively. 
Since $m_t$ and $ν_t$ are almost zero during the initial time steps, they are bias-corrected to $\hat{m_t}$ and $\hat{v_t}$. 
It can be observed that both $\hat{m_t}$ and $\hat{v_t}$ gradually decrease as the time step $t$ increases, which means the role of the first moment estimate starts to weaken, but the role of the second moment estimate starts to increase. 
This is because, as the time step goes on, the parameters in the neural network will be closer to the optimal, and the optimizer should slow down the update pace and focus more on the adjustment of the learning rate of different parameters. 
In the above formulas, the hyper-parameter $β_1$ is the first moment factor, which is usually set to 0.9. 
The hyper-parameter $β_2$ is the second moment factor, which is usually set to 0.999. 
The hyper-parameter $η$ is the learning rate, it is defined as the update step size of the optimizer. 
The hyper-parameter $ε$ is a very small value used to prevent the denominator from being zero. 
And $θ_t$ represents weights and biases at the time step $t$, it will be updated by the overall update vector $\frac{η}{\sqrt{\hat{v_t}+ε}} \hat{m_t}$.


### 13. Mini-batch Training

$$
θ_{t+1} = θ_t - η∇_{θ_t} J(θ_t, x^{(ⅈ:ⅈ+m)})
$$

For a larger dataset, training all the data at the same time is extremely computationally expensive, and neural networks will converge quite slowly. 
Mini-batch training makes the training process more efficient. 
This method will first randomly shuffle the overall training data and then divides the entire dataset into several sub-data sets, and the size of the sub-data is controlled by a hyper-parameter called batch size $m$. 
In each time step, the neural network only learns one batch of training data $x^{(ⅈ:ⅈ+m)}$, and back-propagates based on the loss generated by this batch of data. 
When the neural network learns the entire training dataset once, a.k.a. one epoch has been completed, the parameters in the neural network have been updated many times. 
This is why mini-batch training can help the neural network converge faster. 
In the above formula, the hyper-parameter $η$ is the learning rate, it is defined as the update step size of the optimizer. 
And $θ_t$ represents weights and biases at the time step $t$, it will be updated by the overall update vector $η∇_{θ_t} J(θ_t, x^{(ⅈ:ⅈ+m)})$.











