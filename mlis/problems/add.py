# In this task, each input consists of
# a length-n sequence of depth 2, with all values randomly
# chosen in [0, 1], and the second dimension being all zeros
# except for two elements that are marked by 1. The objective
# is to sum the two random values whose second dimensions are marked by 1.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import math
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = 32
        self.linear1 = nn.Linear(self.input_size * 2, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def calc_loss(self, output, target):
        loss = F.nll_loss(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.data.max(1, keepdim=True)[1]
        return predict

class Solution():
    def __init__(self):
        self.learning_rate = 0.1

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        while True:
            data = train_data
            target = train_target
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                return step
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # calculate loss
            loss = model.calc_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
            with torch.no_grad():
                # get the index of the max probability
                predict = model.calc_predict(output)
                # Number of correct predictions
                correct = target.eq(predict.data.view_as(target)).long().sum().item()
                # Total number of needed predictions
                total = target.data.size(0)
                # print progress of the learning
                self.print_stats(step, loss, correct, total)
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
        self.test_limit = 0.9

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, seed):
        torch.manual_seed(seed)
        data = torch.ByteTensor(data_size, input_size, 2)
        data[:,:,0].random_(0, 2)
        data[:,:,1].zero_()
        index = torch.LongTensor(data_size, 2)
        first_index = torch.LongTensor(data_size).random_(0, input_size)
        second_index = torch.LongTensor(data_size).random_(0, input_size-1)
        second_index = second_index + (second_index >= first_index).long()
        offset = torch.arange(0, data.size(0)*data.size(1), data.size(1), out=torch.LongTensor())
        data[:,:,1].view(-1)[first_index+offset] = 1
        data[:,:,1].view(-1)[second_index+offset] = 1
        target = (data[:,:,0]*data[:,:,1]).sum(dim=1)
        return (data.float(), target.long())

    def create_case_data(self, case):
        input_size = 10*case
        data_size = 256*32

        data, target = self.create_data(2*data_size, input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} length of sequence".format(input_size)).set_input_size(input_size)

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
