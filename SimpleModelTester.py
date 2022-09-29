# Thsi model tester only includes models with two layers

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class SSTDataset(Dataset):
    
    def __init__(self):
        # Data loading
        dataAll = pd.read_csv('~/ForDarwin7/MinimizedPerfectedData/PerfectGlobalDataset.csv')
        CORELABELS = dataAll.iloc[:, 0]
        dataAllY = dataAll.iloc[:, [30]] # This used to contain the core row (row 0)
        print(dataAllY)
        dataAllX = dataAll.iloc[:, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
        # Reshape to testing and training sets:
        all_x = torch.tensor(dataAllX.to_numpy(), dtype=torch.float32)
        all_y = torch.tensor(dataAllY.to_numpy(), dtype=torch.float32)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(all_x, all_y, test_size=0.2)

        self.n_samples = self.x_train.shape[0]
        self.n_features_x = self.x_train.shape[1]
        self.n_features_y = self.y_train.shape[1]
        
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    
    def __len__(self):
        return self.n_samples
    
    def features(self):
        return self.n_features_x, self.n_features_y
    
    def testSets(self):
        return self.x_test, self.y_test
    
class LinModel(nn.Module):
    
    def __init__(self, n_input_dim, n_output_dim, hidden_size1, act1_, hidden_size2, act2_, hidden_size3, act3_, hidden_size4, act4_):
        super(LinModel, self).__init__()
        
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0
        self.l4 = 0
        self.act1 = 0
        self.act2 = 0
        self.act3 = 0
        self.act4 = 0
        
        if act1_ == 'Li':
            self.act1 = nn.ReLU()
        elif act1_ == 'Lo':
            self.act1 = nn.LogSigmoid()
        elif act1_ == 'T':
            self.act1 = nn.Tanh()
        else:
            self.act1 = 0
        if hidden_size1 == 0:
            self.l1 = nn.Linear(n_input_dim, n_output_dim)
        else:
            self.l1 = nn.Linear(n_input_dim, hidden_size1)
            if hidden_size2 == 0:
                self.l2 = nn.Linear(hidden_size1, n_output_dim)
            else: 
                self.l2 = nn.Linear(hidden_size1, hidden_size2)
            
                if act2_ == 'Li':
                    self.act2 = nn.ReLU()
                elif act2_ == 'Lo':
                    self.act2 = nn.LogSigmoid()
                elif act2_ == 'T':
                    self.act2 = nn.Tanh()
                else:
                    self.act2 = 0
                
                if hidden_size3 == 0:
                    self.l3 = nn.Linear(hidden_size2, n_output_dim)
                else:
                    self.l3 = nn.Linear(hidden_size2, hidden_size3)
                
                    if act3_ == 'Li':
                        self.act3 = nn.ReLU()
                    elif act3_ == 'Lo':
                        self.act3 = nn.LogSigmoid()
                    elif act3_ == 'T':
                        self.act3 = nn.Tanh()
                    else:
                        self.act3 = 0
                    
                    if hidden_size4 == 0:
                        self.l4 = nn.Linear(hidden_size3, n_output_dim)
                    else:
                        self.l4 = nn.Linear(hidden_size3, n_output_dim) # We can include another layer if we want
                
                        if act4_ == 'Li':
                            self.act4 = nn.ReLU()
                        elif act4_ == 'Lo':
                            self.act4 = nn.LogSigmoid()
                        elif act4_ == 'T':
                            self.act4 = nn.Tanh()
                        else:
                            self.act4 = 0                       
            
        
    def forward(self, x):
        out = self.l1(x)
        if self.act1 != 0:
            out = self.act1(out)
        if self.l2 != 0:
            out = self.l2(out)
            if self.act2 != 0:
                out = self.act2(out)
            if self.l3 != 0:
                out = self.l3(out)
                if self.act3 != 0:
                    out = self.act3(out)
                if self.l4 != 0:
                    out = self.l4(out)
                    if self.act4 != 0:
                        out = self.act4(out)

        return out

    
# This function should take in a bunch of hyperparameters and architectural parameters and output a training and testing RMSE
# include whether to normalize the input or output later!!! or now? not now
def trialRun(loader, input_size_, output_size_, num_epochs_ , learning_rate_ , neurons_layer1_ , activation_layer1_ , neurons_layer2_ , activation_layer2_ , neurons_layer3_ , activation_layer3_ , neurons_layer4_ , activation_layer4_):
    
    # Now take all variables:
    dataloader = loader
    dataiter = iter(dataloader) # To iterate through the entire data set
    input_size = input_size_
    output_size = output_size_
    learning_rate = learning_rate_
    num_epochs = num_epochs_
    #batch_size = batch_size_
    n_iters = math.ceil(total_samples / batch_size)
    layer1 = neurons_layer1_
    activation1 = activation_layer1_
    layer2 = neurons_layer2_
    activation2 = activation_layer2_
    layer3 = neurons_layer3_
    activation3 = activation_layer3_
    layer4 = neurons_layer4_
    activation4 = activation_layer4_
    
    # Now construct the network:
    # Model input: (self, n_input_dim, n_output_dim, hidden_size1, act1_, hidden_size2, act2_, hidden_size3, act3_, hidden_size4, act4_)
    model = LinModel(input_size, output_size, layer1, activation1, layer2, activation2, layer3, activation3, layer4, activation4)
    
    # Construct loss and optimizer functions (DO NOT VARY)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # TRAINING LOOP (DOES NOT VARY)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Forward, backward, iterate
            # Forward pass - compute prediction
            y_pred = model(inputs)
            loss = torch.sqrt(criterion(labels, y_pred)) # They are the same size
    
            # Backward pass: gradients
            # Zero gradients:
            optimizer.zero_grad()
            loss.backward() # dl/dw
            # Update weights
            optimizer.step()
    
    x_test, y_test = dataset.testSets()
    y_test_pred = model(x_test)
    return loss, torch.sqrt(criterion(y_test, y_test_pred)) # Returns training RMSE, testing RMSE

    

######################################
### ITERATE THROUGH ALL PARAMETERS ###
######################################
epoch_choices = [2500,5000]
layer1_sizes = [64] # Possible layer sizes are in powers of two, including a size of 25 since there are about 25 inputs
layer2_sizes = [8,16]
layer3_sizes = [4,8,16,32,64]
activations1 = ['T'] # Possible activation functions for layer 1(0 meaning no activation)
activations2 = ['Li', 'Lo']
activations3 = ['Li', 'Lo', 'T']
batch_sizes  = [16,32] # Possible batch sizes are powers of two
learning_rates = [0.005] # Should this be 0.01?



# Create the dataloader based on given batch size
results = pd.DataFrame(columns = ['Trial', 'Batch Size', 'Epochs', 'Learning Rate', 'Layer 1 Size', 'Activation 1','Layer 2 Size', 'Activation 2', 'Layer 3 Size', 'Activation 3','Layer 4 Size', 'Activation 4', 'Train RMSE', 'Test RMSE'])
for trial in range(10):
    # We define the dataset here so that it does not change
    # Start by initiating the dataset on the ocean SSTs
    dataset = SSTDataset()
    total_samples = len(dataset)
    input_size, output_size = dataset.features()
    for batch_size in batch_sizes:
        dataloader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=True)
        for epoch_choice in epoch_choices:
            for learn_rate in learning_rates:
                #train, test = trialRun(dataloader, input_size, output_size, epoch_choice, learn_rate, 0, 0, 0, 0, 0, 0, 0, 0)
                #print(f'Trial: {trial}, Batch Size: {batch_size}, Epochs: {epoch_choice}, Learning Rate: {learn_rate}, Train RMSE: {train:.8f}, Test RMSE: {test:.8f}')
                for layer1 in layer1_sizes:
                    for act1 in activations1:
                        #train, test = trialRun(dataloader, input_size, output_size, epoch_choice, learn_rate, layer1, act1, 0, 0, 0, 0, 0, 0)
                        #results = results.append({'Trial' : trial, 'Batch Size' : batch_size, 'Epochs' : epoch_choice, 'Learning Rate' : learn_rate, 'Layer 1 Size' : layer1, 'Activation 1' : act1, 'Layer 2 Size' : 0, 'Activation 2' : 0, 'Layer 3 Size' : 0, 'Activation 3' : 0, 'Layer 4 Size' : 0, 'Activation 4'  : 0, 'Train RMSE' : train, 'Test RMSE' : test}, ignore_index = True)
                        #print(f'Trial: {trial}, Batch Size: {batch_size}, Epochs: {epoch_choice}, Learning Rate: {learn_rate}, Layer 1 Size: {layer1}, Activation 1: {act1}, Train RMSE: {train:.8f}, Test RMSE: {test:.8f}')
                        for layer2 in layer2_sizes:
                            for act2 in activations2:
                                train, test = trialRun(dataloader, input_size, output_size, epoch_choice, learn_rate, layer1, act1, layer2, act2, 0, 0, 0, 0)
                                results = results.append({'Trial' : trial, 'Batch Size' : batch_size, 'Epochs' : epoch_choice, 'Learning Rate' : learn_rate, 'Layer 1 Size' : layer1, 'Activation 1' : act1, 'Layer 2 Size' : layer2, 'Activation 2' : act2, 'Layer 3 Size' : 0, 'Activation 3' : 0, 'Layer 4 Size' : 0, 'Activation 4'  : 0, 'Train RMSE' : train, 'Test RMSE' : test}, ignore_index = True)
                                print(f'Trial: {trial}, Batch Size: {batch_size}, Epochs: {epoch_choice}, Learning Rate: {learn_rate}, Layer 1 Size: {layer1}, Activation 1: {act1}, Layer 2 Size: {layer2}, Activation 2: {act2}, Train RMSE: {train:.8f}, Test RMSE: {test:.8f}')
                                for layer3 in layer3_sizes:
                                    for act3 in activations3:
                                        train, test = trialRun(dataloader, input_size, output_size, epoch_choice, learn_rate, layer1, act1, layer2, act2, layer3, act3, 0, 0)
                                        results = results.append({'Trial' : trial, 'Batch Size' : batch_size, 'Epochs' : epoch_choice, 'Learning Rate' : learn_rate, 'Layer 1 Size' : layer1, 'Activation 1' : act1, 'Layer 2 Size' : layer2, 'Activation 2' : act2, 'Layer 3 Size' : layer3, 'Activation 3' : act3, 'Layer 4 Size' : 0, 'Activation 4'  : 0, 'Train RMSE' : train, 'Test RMSE' : test}, ignore_index = True)
                                        print(f'Trial: {trial}, Batch Size: {batch_size}, Epochs: {epoch_choice}, Learning Rate: {learn_rate}, Layer 1 Size: {layer1}, Activation 1: {act1}, Layer 2 Size: {layer2}, Activation 2: {act2}, Layer 3 Size: {layer3}, Activation 3: {act3}, Train RMSE: {train:.8f}, Test RMSE: {test:.8f}')
#                                         for layer4 in layer_sizes:
#                                             for act4 in activations:
#                                                 train, test = trialRun(dataloader, input_size, output_size, epoch_choice, learn_rate, layer1, act1, layer2, act2, layer3, act3, layer4, act4)
#                                                 results = results.append({'Trial' : trial, 'Batch Size' : batch_size, 'Epochs' : epoch_choice, 'Learning Rate' : learn_rate, 'Layer 1 Size' : layer1, 'Activation 1' : act1, 'Layer 2 Size' : layer2, 'Activation 2' : act2, 'Layer 3 Size' : layer3, 'Activation 3' : act3, 'Layer 4 Size' : layer4, 'Activation 4'  : act4, 'Train RMSE' : train, 'Test RMSE' : test}, ignore_index = True)
#                                                 print(f'Trial: {trial}, Batch Size: {batch_size}, Epochs: {epoch_choice}, Learning Rate: {learn_rate}, Layer 1 Size: {layer1}, Activation 1: {act1}, Layer 2 Size: {layer2}, Activation 2: {act2}, Layer 3 Size: {layer3}, Activation 3: {act3}, Layer 4 Size: {layer4}, Activation 4: {act4}, Train RMSE: {train:.8f}, Test RMSE: {test:.8f}')
# Now we take all of this data to a CSV file
results.to_csv('Test4.csv')
