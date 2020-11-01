import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("Advertising.csv")
x = dataframe.values[:, 2]
y = dataframe.values[:, 4]
plt.scatter(x,y,marker = "o")


def predict(new_ratio, weight, bias):
    return new_ratio*weight + bias

def cost_function(x,y, weight, bias):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight*x[i] + bias))**2
    return sum_error/n 

def update_weight(x,y,weight,bias, learning_rate):
    n= len(x)  
    weight_temp = 0.0 # weigth_temp/n la dao ham theo weight
    bias_temp = 0.0 # bias/n la dao ham theo bias
    for i in range(n):
        weight_temp += -2*x[i]*(y[i]  - (x[i]*weight + bias)) # dao ham ham cost theo weight
        bias_temp += -2*(y[i]  - (x[i]*weight + bias)) # dao ham ham cost theo bias
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias/n)*learning_rate

    return weight, bias


def train(x,y,weight_start, bias_start, learning_rate, iter):
    cos_his = []
    for i in range(iter):
        weight, bias = update_weight(x,y,weight_start, bias_start, learning_rate)
        cost = cost_function(x,y,weight, bias)
        cos_his.append(cost)

    return weight, bias

weight,bias = train(x,y,0.03,0.0014,0.001,1000)
print(weight)
print(bias)
b = np.linspace(-5,50,100)
a = weight*b + bias
plt.plot(b,a,'-r', label='ss') 
plt.show()