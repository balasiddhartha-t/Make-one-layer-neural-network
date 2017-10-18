from numpy import exp,array,random,dot

class NeuralNetwork(object):
 	def __init__(self):
 		#seed the random number so that it generates the same random number
 		#Every time the program runs
 		#Debugger mode on :P
 		random.seed(1)

 		#we model the single neuron with 3 input connections and 1 output connection
 		#we assign random weights to 3x1 matrix, with values in the range -1 to 1
 		#and mean 0
 		self.synaptic_weights = 2 * random.random((3,1)) - 1

 	def __sigmoid(self, x):
 		return 1/(1+exp(-x))

 	#the derivative of the sigmoid is the gradient of the sigmoid curve

 	def __sigmoid_derivative(self, x):
 		return x * (1 - x)
 	
 	#let met adjust the training weights each time, 
 	def train(self, training_set_inputs,training_set_outputs,number_of_training_iterations):
 		for iteration in range(number_of_training_iterations):
 			output = self.think(training_set_inputs)

 			error = training_set_outputs - output

 			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

 			self.synaptic_weights += adjustment

 	def think(self,inputs):
 		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

	#Initialise a Single neuron neural network
	neural_network = NeuralNetwork()

	print( "Random starting synaptic weights:")
	print (neural_network.synaptic_weights)

	#The training set, we have four examples, each consisting of three input values 
	#1 output value
	training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T
	#The .T method represents the transpose matrix of the given array

	#Train the neural network using the traininig set
	#Do it 10,000 times and make small adjustments each times
	neural_network.train(training_set_inputs,training_set_outputs,10000)

	print ("New synaptic weights after training")
	print (neural_network.synaptic_weights)

	#Test the neural network with a new situation
	print ("considering the new situation")
	print (neural_network.think(array([1,0,0])))