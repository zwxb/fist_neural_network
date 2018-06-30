import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# print(rides.head())


rides[:24*10].plot(x='dteday',y='cnt')
# plt.show()
dummy_fields =['season','weathersit','mnth','hr','weekday']
for each in dummy_fields:
    # print(rides[each])
    dummies = pd.get_dummies(rides[each],prefix=each,drop_first=False)
    rides = pd.concat([rides,dummies],axis=1)
fields_to_drop=['instant','dteday','season','weathersit','weekday','atemp','mnth','workingday','hr']
data = rides.drop(fields_to_drop,axis=1)
# print(data.head())


quant_features = ['casual','registered','cnt','temp','hum','windspeed']
scaled_features={}
for each in quant_features:
    mean,std = data[each].mean(),data[each].std()
    scaled_features[each] = [mean,std]
    data.loc[:,each] = (data[each] -mean)/std

test_data = data[-21*24:]
data = data[:-21*24]
target_fields = ['cnt','casual','registered']

features, targets = data.drop(target_fields,axis=1),data[target_fields]
test_features, test_targets = test_data.drop(target_fields,axis=1),test_data[target_fields]


train_features, train_targets = features[:-60*24],targets[:-60*24]
val_features, val_targets = features[-60*24:],targets[-60*24:]

class NeuralNetwork(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes,lr):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0.0,self.input_nodes**-0.5,(self.input_nodes,self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0,self.hidden_nodes**-0.5,(self.hidden_nodes,self.output_nodes))
        self.lr = lr

        self.activation_function = lambda x : 1/(np.exp(x))

    def train(self,features,targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_i_o = np.zeros(self.weights_hidden_to_output.shape)
        for x,y in zip(features,targets):
            hidden_inputs = np.dot(x,self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output)
            final_outputs = final_inputs

            error = y - final_outputs







