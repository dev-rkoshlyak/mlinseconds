# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('./../utils')
import solutionmanager as sm

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = 32
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.sigmoid(x)
        return x

class Solution():
    def __init__(self):
        self = self

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size)

    # Return number of steps used
    def train_model(self, time_limit, model, train_data, train_target):
        step = 0
        # Put model in train mode
        model.train()
        while True:
            time_left = time_limit - time.time()
            # No more time left, stop training
            if time_left < 0.1:
                break
            optimizer = optim.SGD(model.parameters(), lr=1)
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate loss
            bce_loss = nn.BCELoss()
            loss = bce_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step
    
    def print_stats(self, step, loss, correct, total):
        if step % 100 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        random.seed(seed)
        function_size = 1 << input_size
        function_table = [random.randint(0, 1) for _ in range(function_size)]
        total_input_size = input_size+random_input_size
        input_bit_indexes = {x.item():(1<<i) for i,x in enumerate(torch.randperm(total_input_size)[:input_size])}
        data = torch.FloatTensor(data_size, total_input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            fun_ind = i%len(function_table)
            fun_value = function_table[fun_ind]
            for j in range(total_input_size):
                input_bit = random.randint(0, 1)
                if j in input_bit_indexes:
                    input_bit = fun_ind&1
                    fun_ind = fun_ind >> 1
                data[i,j] = float(input_bit)
            target[i] = float(fun_value)
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
