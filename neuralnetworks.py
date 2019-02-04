import numpy as np

class NeuralNetwork(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learnrate):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes


        
        #Initialize weights matrix or by layers
        self.weights_input_to_hidden=np.random.normal(0.0,self.input_nodes**0.5,(self.input_nodes,self.hidden_nodes))
        self.weights_hidden_to_output=np.random.normal(0.0,self.hidden_nodes**0.5,(self.hidden_nodes.self.output_nodes))

        self.lr=learnrate


        #Different Activation function definitions
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        def tanh(x):
            return tanh(x)

        def relu(x):
            return x

        def softmax(x):
            return x


        #self.activation_function=activation
        #By default using only sigmoid
        
        self.activation_function = sigmoid


        
    def train(self,features,target):
        n_records=features.shape[0]
        delta_weights_i_h=np.zeroes(self.weights_input_to_hidden.shape)
        delta_weights_h_o=np.zeroes(self.weights_hidden_to_output.shape)

        for X,y in zip(features,target):
            final_output,hidden_output=self.forward_pass(X)
            delta_weights_i_h, delta_weights_h_o=self.backpropagation(final_output,hidden_output,X,y,delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass(self,X):
        hidden_inputs=np.dot(X,self.weights_input_to_hidden)
        hidden_output=self.activation_function(hidden_inputs)

        final_output=np.dot(hidden_output,self.weights_hidden_to_output)
        final_output=final_output  #NO activation function

        
        return final_outputs, hidden_outputs

    
    def backpropagation(self,final_output,hidden_output,X,y,delta_weights_i_h, delta_weights_h_o):

        error=y-final_output

        hidden_error=np.dot(self.weights_hidden_to_output,error)
        
        output_error_term=error*hidden_outputs*(1-hidden_outputs)

        hidden_error_term=hidden_error*hidden_outputs*(1-hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h=delta_weights_i_h+hidden_error_term * X[:,None]


        # Weight step (hidden to output)
        delta_weights_h_o=delta_weights_h_o+output_error_term* X[:,None]

        return delta_weights_i_h, delta_weights_h_o

    
    def update_weights(self,delta_weights_i_h, delta_weights_h_o, n_records):

        self.weights_hidden_to_output=self.weights_hidden_to_output+delta_weights_h_o*self.lr/n_records
        self.weights_input_to_hidden=self.weights_input_to_hidden+(delta_weights_i_h*self.lr/n_records)

    def run(self,features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''

        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_output=self.activation_funcction(hidden_inputs)

        final_input=np.dot(hidden_output,self.weights_hidden_to_output)
        final_output=final_input

        return final_output


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 15000
learning_rate = 0.2
hidden_nodes = 8
output_nodes = 1

