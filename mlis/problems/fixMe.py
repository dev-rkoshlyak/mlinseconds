# There are solution for generalCpu in 10 steps.
# 
# BUT unfortunetly, someone made mistake and disabled batch normalization.
# See "FIX ME"
#
# You can not simple fix mistake, you can change only activation function at this point
import math
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

# Note: activation function should be element independent
# See: check_independence method
class MyActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

###
###
### Don't change code after this line
###
###
class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.do_norm = solution.do_norm
        layers_size = [input_size] + [solution.hidden_size]*solution.layers_number + [output_size]
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(layers_size, layers_size[1:])])
        if self.do_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(a, track_running_stats=False) for a in layers_size[1:]])

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.do_norm:
                x = self.batch_norms[i](x)
            if i != len(self.linears)-1:
                x = MyActivation.apply(x)
            else:
                x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        loss = nn.BCELoss()
        return loss(output, target)

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self.learning_rate = 0.05
        self.momentum = 0.8
        self.layers_number = 3
        self.hidden_size = 30
        # FIX ME:) But you can change only activation function
        self.do_norm = False
        self.layers_number = 8
        self.learning_rate_grid = [0.005, 0.05, 0.5]
        self.momentum_grid = [0.7, 0.8, 0.9]
        #self.layers_number_grid = [1,2,3,4,5,6,7,8]
        #self.hidden_size_grid = [20, 30, 40]
        #self.do_norm_grid = [True, False]
        self.iter = 0
        self.iter_number = 100
        self.grid_search = None

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # activation should be independent
    def check_independence(self):
        ind_size = 100
        for i in range(ind_size+1):
            x = torch.FloatTensor(ind_size).uniform_(-10, 10)
            same = MyActivation.apply(x)[:i] == MyActivation.apply(x)[:i]
            assert same.long().sum() == i, "Independent function only"

    # Return trained model
    def train_model(self, train_data, train_target, context):
        self.check_independence()
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total:
                break
            # calculate loss
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()

        if self.grid_search:
            self.grid_search.add_result('step', context.step)
            if self.iter == self.iter_number-1:
                print(self.grid_search.choice_str, self.grid_search.get_stats('step'))
        return model
    
    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=100, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
