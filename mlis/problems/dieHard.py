# There are 2 functions defined from input. One easy one and one hard one.
# On training data easy and hard functions produce same result and on
# test data you need to predict hard function.
# Easy function - depends on fewer inputs.
# Hard function - depends on more inputs.
# Easy and hard function depends on different inputs.
# Functions is a random functions of n-inputs, it's guarantee that
# functions depends on n inputs.
# For example:
# Inputs:
# x0, x1, x2, x3, x4, x5, x6, x7
# Easy function:
# x0^x1
# Hard function:
# x2^x3^x4^x5^x6^x7
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class Solution():
    # Return trained model
    def train_model(self, train_data, train_target, context):
        print("See helloXor for solution template")
        exit(0)

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 0.75

class DataProvider:
    def __init__(self):
        self.number_of_cases = 20

    def full_func(self, input_size):
        while True:
            table = torch.ByteTensor(1<<input_size).random_(0, 2)
            vals = torch.ByteTensor(input_size, 2).zero_()
            depend_count = 0
            for i in range(input_size):
                for ind in range(1<<input_size):
                    if table[ind].item() != table[ind^(1<<i)].item():
                        depend_count += 1
                        break
            if depend_count == input_size:
                return table

    def tensor_to_int(self, tensor):
        tensor = tensor.view(-1)
        res = 0
        for x in tensor:
            res = (res<<1)+x.item()
        return res

    def int_to_tensor(self, ind, tensor):
        for i in range(tensor.size(0)):
            tensor[i] = (ind >> i)&1

    def create_data(self, seed, easy_table, hard_table, easy_input_size, hard_input_size, easy_correct):
        input_size = easy_input_size + hard_input_size
        data_size = 1 << input_size
        data = torch.ByteTensor(data_size, input_size)
        target = torch.ByteTensor(data_size, 1)
        count = 0
        for ind in range(data_size):
            self.int_to_tensor(ind, data[count])
            easy_ind = ind & ((1 << easy_input_size)-1)
            hard_ind = ind >> easy_input_size
            easy_value = easy_table[easy_ind].item()
            hard_value = hard_table[hard_ind].item()
            target[count, 0] = hard_value
            if not easy_correct or easy_value == hard_value:
                count += 1
        data = data[:count,:]
        target = target[:count,:]
        perm = torch.randperm(count)
        data = data[perm]
        target = target[perm]
        return (data.float(), target.float())

    def create_case_data(self, case):
        easy_input_size = 2
        hard_input_size = 6

        random.seed(case)
        torch.manual_seed(case)
        easy_table = self.full_func(easy_input_size)
        hard_table = self.full_func(hard_input_size)
        train_data, train_target = self.create_data(case, easy_table, hard_table, easy_input_size, hard_input_size, True)
        test_data, test_target = self.create_data(case, easy_table, hard_table, easy_input_size, hard_input_size, False)
        perm = torch.randperm(train_data.size(1))
        train_data = train_data[:,perm]
        test_data = test_data[:,perm]
        return sm.CaseData(case, Limits(), (train_data, train_target), (test_data, test_target)).set_description("Easy {} inputs and hard {} inputs".format(easy_input_size, hard_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
