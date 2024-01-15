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
