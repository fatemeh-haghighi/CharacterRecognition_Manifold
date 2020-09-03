# Neural Networks
# Bonus Assignment

##############################      Main      #################################

def read_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [int(x) * 2 - 1 for x in line if x != "\n"]
        training_data_list.extend([line[:]])
    return training_data_list




###############                 Training                     ##################
 

###############          Enter your code below ...           ##################
def function(value):
    if value >= 0:
        return 1
    else:
        return -1

class Perceptron_neuron():
    def __init__(self, weights, a, bias, epoch, function):
        self.weights = weights
        self.bias = bias
        self.a = a
        self.out = 0
        self.epoch = epoch
        self.update_counter = 0
        self.function = function
        self.fit_value = 0
    
    def get_counter(self):
        return self.update_counter
    
    def update_rule(self, inputs, t):
        self.update_counter += 1
        for i in range(len(inputs)):
            h = self.get_h_value(inputs)
            if h - t != 0:
                self.bias = self.bias + self.a * t
                # print("*************",type(inputs[i])
                temp = self.a * inputs[i] *t
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] + temp

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias

    def get_fit(self):
        return self.fit_value
    
    def get_net_value(self, inputs):
        out = 0
        for i in range(len(inputs)):
            out += self.weights[i] * inputs[i]
        out += self.bias
        self.out = out
        return out

    def get_h_value(self, instance):
        return self.function(self.get_net_value(instance))

    def fit(self, inputs, target):
        if self.update_counter >= self.epoch:
            return 1
        else:
            if target == self.get_h_value(inputs):
                self.fit_value = 1
                return 1
            else:
                return 0


epoch = 4
a = 0.005

training_data_list = read_file(file="OCR_train.txt")
for i in range(len(training_data_list)):
    train_error = 0

    print("\ntrainning part in number ", i)
    inputs = training_data_list[i][0:63]

    target = training_data_list[i][64: ]
    weights = []
    for i in range(len(inputs)):
        weights.append(0)

    
    neuron1 = Perceptron_neuron(weights, a, 1, epoch, function)
    while(neuron1.fit(inputs, target[0]) != 1):
        neuron1.update_rule(inputs, target[0])
    counter = neuron1.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron1")
    if(neuron1.get_fit() == 0):
        train_error += 1


    neuron2 = Perceptron_neuron(weights, a, 1, epoch, function)  
    while(neuron2.fit(inputs, target[1]) != 1):
        neuron2.update_rule(inputs, target[1])  
    counter = neuron2.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron2")
    if(neuron2.get_fit() == 0):
        train_error += 1



    neuron3 = Perceptron_neuron(weights, a, 1, epoch, function)    
    while(neuron3.fit(inputs, target[2]) != 1):
        neuron3.update_rule(inputs, target[2])
    counter = neuron3.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron3")
    if(neuron3.get_fit() == 0):
        train_error += 1


    neuron4 = Perceptron_neuron(weights, a, 1, epoch, function)
    while(neuron4.fit(inputs, target[3]) != 1):
        neuron4.update_rule(inputs, target[3])  
    counter = neuron4.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron4") 
    if(neuron4.get_fit() == 0): 
        train_error += 1



    neuron5 = Perceptron_neuron(weights, a, 1, epoch, function) 
    while(neuron5.fit(inputs, target[4]) != 1):
        neuron5.update_rule(inputs, target[4])   
    counter = neuron5.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron5")
    if(neuron5.get_fit() == 0):
        train_error += 1



    neuron6 = Perceptron_neuron(weights, a, 1, epoch, function) 
    while(neuron6.fit(inputs, target[5]) != 1):
        neuron6.update_rule(inputs, target[5])   
    counter = neuron6.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron6")
    if(neuron6.get_fit() == 0):
        train_error += 1



    neuron7 = Perceptron_neuron(weights, a, 1, epoch, function)    
    while(neuron7.fit(inputs, target[6]) != 1):
        neuron7.update_rule(inputs, target[6])
    counter = neuron7.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron7")
    if(neuron7.get_fit() == 0):
        train_error += 1

    
    if(train_error != 0):
        print("fault accured during training")
    else:
        print("successfull")
    print("------------------------------------------")








###############          Enter your code above ...           ##################
    


# print("\nThe Neural Network has been trained in " + str(epoch) + " epochs.")



###############                   Testing                    ##################



###############          Enter your code below ...           ##################

testing_data_list = read_file(file="OCR_test.txt")
_total = 0
_error = 0
for i in range(len(testing_data_list)):
    _total += 1
    test_error = 0

    print("\ntest part in number ", i)
    inputs = testing_data_list[i][0:63]

    target = testing_data_list[i][64: ]
    weights = []
    for i in range(len(inputs)):
        weights.append(0)

    # epoch = 2
    # a = 0.01
    neuron1 = Perceptron_neuron(weights, a, 1, epoch, function)
    while(neuron1.fit(inputs, target[0]) != 1):
        neuron1.update_rule(inputs, target[0])
    counter = neuron1.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron1")
    if(neuron1.get_fit() == 0):
        test_error += 1


    neuron2 = Perceptron_neuron(weights, a, 1, epoch, function)  
    while(neuron2.fit(inputs, target[1]) != 1):
        neuron2.update_rule(inputs, target[1])  
    counter = neuron2.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron2")
    if(neuron2.get_fit() == 0):
        test_error += 1



    neuron3 = Perceptron_neuron(weights, a, 1, epoch, function)    
    while(neuron3.fit(inputs, target[2]) != 1):
        neuron3.update_rule(inputs, target[2])
    counter = neuron3.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron3")
    if(neuron3.get_fit() == 0):
        test_error += 1


    neuron4 = Perceptron_neuron(weights, a, 1, epoch, function)
    while(neuron4.fit(inputs, target[3]) != 1):
        neuron4.update_rule(inputs, target[3])  
    counter = neuron4.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron4") 
    if(neuron4.get_fit() == 0): 
        test_error += 1



    neuron5 = Perceptron_neuron(weights, a, 1, epoch, function) 
    while(neuron5.fit(inputs, target[4]) != 1):
        neuron5.update_rule(inputs, target[4])   
    counter = neuron5.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron5")
    if(neuron5.get_fit() == 0):
        test_error += 1



    neuron6 = Perceptron_neuron(weights, a, 1, epoch, function) 
    while(neuron6.fit(inputs, target[5]) != 1):
        neuron6.update_rule(inputs, target[5])   
    counter = neuron6.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron6")
    if(neuron6.get_fit() == 0):
        test_error += 1



    neuron7 = Perceptron_neuron(weights, a, 1, epoch, function)    
    while(neuron7.fit(inputs, target[6]) != 1):
        neuron7.update_rule(inputs, target[6])
    counter = neuron7.get_counter()
    # if(counter < epoch):
        # print("reach to target before epoch in neuron7")
    if(neuron7.get_fit() == 0):
        test_error += 1

    
    if(test_error != 0):
        print("fault accured during testing")
        _error += 1
    print("------------------------------------------")
    










###############          Enter your code above ...           ##################

print("\n\nPercent of Error in NN: " + str(_error / _total))













