import numpy as np
import matplotlib.pyplot as plt
import random

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
    
    def update_rule(self, inputs, t):
        self.update_counter += 1
        for i in range(len(inputs)):
            h = self.get_h_value(inputs[i])
            if h - t[i] != 0:
                self.bias = self.bias + self.a * t[i]
                self.weights = self.weights + self.a * inputs[i] *t[i]

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
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
            output = 0
            for instance in inputs:
                output += self.get_net_value(instance)
            if self.get_h_value(output) - target == 0:
                return 1
            else:
                return 0


x_array = []
y_array = []
r_array = []

def is_unique(x_array, y_array, r_array, x, y, r):
    out = []
    if(len(x_array) == 0):
        out.append(1)
    for i in range(len(x_array)):
        if np.sqrt((x_array[i] - x) ** 2 + (y_array[i] - y) ** 2) < r + r_array[i]:
            out.append(0)
        else:
            out.append(1)
    return out

def make_circle(x_array, y_array, r_array, max_x, max_y, max_r):
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    r = random.randint(5, max_r)
    res = []
    for i in range(len(x_array)):
        res.append(0)

    while(True):
        out = is_unique(x_array, y_array, r_array, x, y, r)
        for i in range(len(out)):
            if out[i] == 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                r = random.randint(5, max_r)
        break
    x_array.append(x)
    y_array.append(y)
    r_array.append(r)


model_number = random.randint(3, 5)

def make_all_circles(x_array, y_array, r_array, model_number):
    for i in range(model_number):
        make_circle(x_array, y_array, r_array, 300, 300, 30)

make_all_circles(x_array, y_array, r_array, model_number)

s= []
for i in range(len(r_array)):
    s.append(np.pi * (r_array[i]** 2))

plt.scatter(x_array, y_array, s, color = 'red', cmap='jet')
plt.title('random circles')
plt.show()


def generate_data(x_array, y_array, r_array):
    x_data = []
    y_data = []
    for i in range(len(x_array)):
        for j in range(20):
            temp_r = random.randint(0, r_array[i])
            temp_angle = random.randint(0, 360)
            x = temp_r * np.cos(temp_angle) + x_array[i]
            y = temp_r * np.sin(temp_angle) + y_array[i]
            x_data.append(x)
            y_data.append(y)
    

    return x_data, y_data

x_data, y_data = generate_data(x_array, y_array, r_array)
plt.scatter(x_data, y_data, color = 'red')
plt.title('generated date')  
plt.show()  
target = []
for i in range(model_number):
    temp = []
    for j in range(model_number):  
        temp.append(0)
    temp[i] = 1
    target.append(temp)     

data = []
weights = []
for i in range(len(x_data)):
    data.append([x_data[i], y_data[i], x_data[i]**2, y_data[i]**2])
    weights.append([0, 0, 0, 0])



# print(target)

a = 0.2
bias = 1
epoch = 100
for i in range(model_number):
    for j in range(model_number):
        n = Perceptron_neuron(weights, a, bias, epoch, function)  
        while(n.fit(data, target[i]) != 1):
            n.update_rule(data, target[i][j])   
                
    
