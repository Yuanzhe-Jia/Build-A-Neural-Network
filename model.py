# info: COMP5329 - ASS 1
# study: Build A Neural Network From Scratch
# author: Yuanzhe Jia
# sid：510534297
# date: 2022-04-02

"""
Note: in this model.py file, the author defined all the neural network modules.
those functions will then be import by experimental_document.ipynb file
for neural networks building and evaluation.

In this file, the author realized many modules, including:
- 1. multiple hidden layers;
- 2. initialization: Kaiming Initialization;
- 3. regularization: Weight Decay; Batch Normalization; Dropout;
- 4. activation: ReLU; GELU; Tanh;
- 5. optimizer: Momentum in SGD; Adam;
- 6. loss: Softmax and cross-entropy Loss;
- 7. training: Mini-batch training;

"""

############################# Load Libraries #############################


# the author only use numpy and time packages in this file
import numpy as np
import time


############################## Dense Layers ##############################


class dense():
    
    def __init__(self, input_size, output_size, init=1, weight_decay=0, optimizer=1):
        # input
        self.input = None
        
        # Kaiming initialization (1: ReLU; 2: Tanh; else: Sigmoid, etc.)
        self.init = init
 
        if self.init == 1:  
            self.w = np.random.uniform(
                low = -np.sqrt(6. / input_size),
                high = np.sqrt(6. / input_size),
                size = (input_size, output_size))

        elif self.init == 2:
            self.w = np.random.uniform(
                low = -np.sqrt(6. / input_size) * 5 / 3,
                high = np.sqrt(6. / input_size) * 5 / 3,
                size = (input_size, output_size))
            
        else:
            self.w = np.random.uniform(
                low = -np.sqrt(3. / input_size),
                high = np.sqrt(3. / input_size),
                size = (input_size, output_size))
            
        self.b = np.zeros([1, output_size])

        # weight decay
        self.decay = weight_decay
        
        # optimizer (1: momentum in SGD; 2: Adam)
        self.optimizer = optimizer
        
        # gradients
        self.w_grad = np.zeros([input_size, output_size])
        self.b_grad = np.zeros([1, output_size])
        self.b_grad_list = []
        
        # accumulated gradients for momentum optimizer
        self.w_s = np.zeros([input_size, output_size])
        self.b_s = np.zeros([1, output_size])
        
        # accumulated gradients for Adam optimizer
        self.w_m = np.zeros([input_size, output_size])
        self.b_m = np.zeros([1, output_size])
        
        # accumulated "squared" gradients for Adam optimizer
        self.w_v = np.zeros([input_size, output_size])
        self.b_v = np.zeros([1, output_size])

    def forward_propagation(self, x, train_mode=True):
        # output = weights * intput + biases 
        self.input = x
        output = np.dot(self.input, self.w) + self.b
        return output
    
    def backward_propagation(self, output_error, learning_rate):
        # derivative of weights = input
        # derivative of biases = 1
        self.w_grad = np.dot(self.input.T, output_error)
        self.b_grad = np.sum(output_error)
        self.b_grad_list.append(self.b_grad)
        
        # *** Momentum in SGD ***
        if self.optimizer == 1:
            # s(t) = momentum * s(t-1) + learning_rate * gradients(t)
            self.w_s = 0.9 * self.w_s + learning_rate * self.w_grad
            self.b_s = 0.9 * self.b_s + learning_rate * self.b_grad
                            
            # *** weight decay -- researcher's veiw ***
            # self.w = self.w * (1 - learning_rate * self.decay) - self.w_s
            # self.b = self.b * (1 - learning_rate * self.decay) - self.b_s
            
            # *** weight decay -- engineer's veiw ***
            self.w -= self.w_s
            self.b -= self.b_s
            self.w = self.w * (1 - self.decay)
            self.b = self.b * (1 - self.decay)

        # *** Adam optimizer ***
        if self.optimizer == 2:
            # m(t) = beta1 * m(t-1) + (1-beta1) * gradients(t)
            self.w_m = 0.9 * self.w_m + (1 - 0.9) * self.w_grad
            self.b_m = 0.9 * self.b_m + (1 - 0.9) * self.b_grad
            
            # v(t) = beta2 * v(t-1) + (1-beta2) * gradients(t) ** 2
            self.w_v = 0.999 * self.w_v + (1 - 0.999) * self.w_grad ** 2
            self.b_v = 0.999 * self.b_v + (1 - 0.999) * self.b_grad ** 2
            
            # m_hat(t) = m(t) / (1 - beta1 ** t)
            w_m_hat = self.w_m / (1 - 0.9 ** len(self.b_grad_list))
            b_m_hat = self.b_m / (1 - 0.9 ** len(self.b_grad_list))
            
            # v_hat(t) = v(t) / (1 - beta2 ** t)
            w_v_hat = self.w_v / (1 - 0.999 ** len(self.b_grad_list))
            b_v_hat = self.b_v / (1 - 0.999 ** len(self.b_grad_list))
            
            # *** weight decay -- researcher's veiw ***
            # self.w = self.w * (1 - learning_rate * self.decay) - learning_rate / (np.sqrt(w_v_hat) + 1e-8) * w_m_hat
            # self.b = self.b * (1 - learning_rate * self.decay) - learning_rate / (np.sqrt(b_v_hat) + 1e-8) * b_m_hat
            
            # *** weight decay -- engineer's veiw ***
            self.w -= learning_rate / (np.sqrt(w_v_hat) + 1e-8) * w_m_hat
            self.b -= learning_rate / (np.sqrt(b_v_hat) + 1e-8) * b_m_hat
            self.w = self.w * (1 - self.decay)
            self.b = self.b * (1 - self.decay)
        
        input_error = np.dot(output_error, self.w.T)
        return input_error   
    


########################### Batch Normalisation ###########################


class batch_norm():
    
    def __init__(self, dims, ratio=0.9):
        # params
        self.input = None
        self.input_test = None
        self.input_mean = None
        self.input_var = None
        self.input_norm = None
        self.n = None
        
        # running average ratio
        self.ratio = ratio
        
        # avoid the denominator is zero
        self.eps = 1e-8
        
        # params need be fine-tune
        self.gamma = np.random.rand(dims) - 0.5
        self.beta = np.random.rand(dims) - 0.5
        
        # params used for test data
        self.running_mean = np.zeros(dims)
        self.running_var = np.ones(dims)
    
    def forward_propagation(self, x, train_mode=True):
        
        # consider training mode is "True" or "False"
        # if "True", calculate mean and var for each batch 
        if train_mode == True:
            # get input data
            self.input = x
            
            # get number of input batchs
            self.n = x.shape[0]
    
            # normalising: input => mean => var => norm
            self.input_mean = np.mean(self.input, axis=0)
            self.input_var = np.var(self.input, axis=0)
            self.input_norm = (self.input - self.input_mean) / (np.sqrt(self.input_var + self.eps))
            
            # stretching and moving
            output = self.gamma * self.input_norm + self.beta           
            
            # parms for test
            self.running_mean = self.ratio * self.running_mean + (1 - self.ratio) * self.input_mean
            self.running_var = self.ratio * self.running_var + (1 - self.ratio) * self.input_var
            return output
        
        # if "False", use running mean and running var
        if train_mode == False:

            # get input data
            self.input_test = x
            
            # calculate output
            output = self.gamma * (self.input_test - self.running_mean) / (np.sqrt(self.running_var + self.eps)) + self.beta
            return output
        
    def backward_propagation(self, output_error, learning_rate):
        # calculate params' gradients
        gamma_grad = np.sum(output_error * self.input_norm, axis=0)
        beta_grad = np.sum(output_error, axis=0)
        
        # update params
        self.gamma -= learning_rate * gamma_grad
        self.beta -= learning_rate * beta_grad
        
        # update input_error: norm => var => mean => input
        norm_grad = output_error * self.gamma

        var_grad = (-0.5 * norm_grad * (self.input - self.input_mean) * (self.input_var + self.eps)**-1.5).sum(axis=0)
        mean_grad = (-1 * norm_grad * (self.input_var + self.eps)**0.5).sum(axis=0) + \
                    (-2 * var_grad * (self.input - self.input_mean) / self.n).sum(axis=0)
        input_error = norm_grad / ((self.input_var + self.eps)**0.5) + 2 * var_grad * (self.input - self.input_mean) / self.n + \
                      mean_grad / self.n
        return  input_error
    
    
     
######################### Activation Function #########################


# *** ReLU activation function ***

class relu():
    
    def __init__(self):
        self.input = None
      
    def forward_propagation(self, x, train_mode=True):
        # output = relu(input) = max(0, input)
        self.input = x
        output = np.clip(self.input, 0, np.inf)
        return output
    
    def backward_propagation(self, output_error, learning_rate):
        # derivative = 0 / 1
        input_error = (self.input > 0) * output_error
        return input_error

# *** GELU activation function ***

class gelu():
    
    def __init__(self):
        self.input = None
    
    def forward_propagation(self, x, train_mode=True):
        self.input = x
        output = 0.5 * self.input * (1 + np.tanh(np.sqrt(2/np.pi) * (self.input + 0.044715 * self.input**3)))
        return output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = output_error * (((np.tanh((np.sqrt(2) * (0.044715 * self.input ** 3 + self.input)) / np.sqrt(np.pi)) + \
                                        ((np.sqrt(2) * self.input * (0.134145 * self.input ** 2 + 1) * \
                                          ((1 / np.cosh((np.sqrt(2) * (0.044715 * self.input ** 3 + self.input)) / \
                                                        np.sqrt(np.pi))) ** 2)) /  np.sqrt(np.pi) + 1))) / 2)
        return input_error

# *** Tanh activation function ***

class tanh():
    
    def __init__(self):
        self.input = None
      
    def forward_propagation(self, x, train_mode=True):
        # output = tanh(input) 
        #      = (exp(input) - exp(-input)) / (exp(input) + exp(-input))
        self.input = x
        output = np.tanh(self.input)
        return output
    
    def backward_propagation(self, output_error, learning_rate):
        # derivative = 1 - tanh(input)**2
        input_error = (1 - np.tanh(self.input)**2) * output_error
        return input_error
    


################################ Dropout ################################


class dropout():
    
    def __init__(self, p):
        self.mask = None
        
        # p is the probability of remaining neurons
        self.p = p
    
    def forward_propagation(self, x, train_mode=True):
        self.mask = np.random.binomial(1, self.p, size=x.shape)
        
        # here the author used the inversed dropout - "mask / p"
        # thus, the test process will be easy to implement
        self.mask = np.true_divide(self.mask, self.p)
        output = self.mask * x
        return output
    
    def backward_propagation(self, output_error, learning_rate):
        # derivative = self.mask
        input_error = output_error * self.mask
        return input_error
    
    

######################## Softmax Cross-entropy Loss ########################


def softmax(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

def softmax_cross_entropy_loss(softmax_probs_array, y_onehot):
    # get the predicted probability corresponding to the true label
    indices = np.argmax(y_onehot, axis=1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]

    # Loss(y_hat, y) = - 1/n * sum(y * log(y_hat))
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def softmax_cross_entropy_derivatives(softmax_probs_array, y_onehot):
    # derivative = y_hat - y
    # here y should be the one-hot vectors
    return softmax_probs_array - y_onehot



##################### Evaluation Prediciton Accuracy #####################


def categorical_accuracy(y_pred, y_true):
    # label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] => argmax(label) = 1 => maximum number index
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    accuracy = np.mean((y_true == y_pred).astype(np.float32))
    return accuracy



####################### Structure of Neural Network #######################


class model():
    
    def __init__(self):
        self.layers = []
        self.loss_value = None
        self.loss_derivative = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def loss(self, loss_value, loss_derivative):
        self.loss_value = loss_value
        self.loss_derivative = loss_derivative
    
    def fit(self, train_data, train_label, batch_size, epochs, learning_rate, val_data, val_label, val_mode=False):
        # timing
        start = time.time()
        
        # initialise
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        
        # calculate number of batch loops in each epoch
        train_size = train_data.shape[0]
        if train_size % batch_size == 0:
            batch_num = int(train_size / batch_size)
        else:
            batch_num = int(train_size / batch_size) + 1
        
        # epoch loop
        for i in range(epochs):
            
            # *** mini-batch training ***
            for j in range(batch_num):
                
                output = train_data[j * batch_size : (j+1) * batch_size]
                ground_truth = train_label[j * batch_size : (j+1) * batch_size]
                
                # forward propagation => calculate final output
                for layer in self.layers:
                    output = layer.forward_propagation(output, train_mode=True)
                
                # normalise output 
                softmax_output = softmax(output)
                
                # calculate training loss, gradients
                train_loss = self.loss_value(softmax_output, ground_truth)
                train_error = self.loss_derivative(softmax_output, ground_truth)
   
                # update weights & biases in each layer
                for layer in reversed(self.layers):
                    train_error = layer.backward_propagation(train_error, learning_rate)
            
            # calculate training accuracy            
            x = train_data
            y = train_label
            
            for layer in self.layers:
                x = layer.forward_propagation(x, train_mode=True)
            
            y_hat = softmax(x)
            train_acc = categorical_accuracy(y_hat, y)*100
   
            
            # calculate validation loss, accuracy
            # consider validation model is "True" or "False"
            if val_mode == True:
                val_x = val_data
                val_y = val_label
                
                for layer in self.layers:
                    val_x = layer.forward_propagation(val_x, train_mode=True)
                
                val_y_hat = softmax(val_x)
                val_loss = self.loss_value(val_y_hat, val_y)
                val_acc = categorical_accuracy(val_y_hat, val_y)*100
            
                # print results
                print("epoch:{:>3d} /{:>3d}  =>  train loss = {:.3f};  train acc = {:.2f}%;  val loss = {:.3f};  val acc = {:.2f}%;"
                      .format(i+1, epochs, train_loss, train_acc, val_loss, val_acc))
                
            if val_mode == False:
                val_loss = 0
                val_acc = 0
                
                # print results
                print("epoch:{:>3d} /{:>3d}  =>  train loss = {:.3f};  train acc = {:.2f}%;"
                      .format(i+1, epochs, train_loss, train_acc))
            
                    
            # update training history
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)            
        
        # timing
        end = time.time()
        print()
        print("total training time: {:.2f}s".format(end-start))
        return train_loss_list, train_acc_list, val_loss_list, val_acc_list
    
    def predict(self, x_test, train_mode=False):
        # save predictions
        predict = []
        
        for i in range(len(x_test)):
            
            # forward propagation for test data
            output = x_test[i]
            for layer in self.layers:
                output = layer.forward_propagation(output, train_mode)
            predict.append(output)
        return predict