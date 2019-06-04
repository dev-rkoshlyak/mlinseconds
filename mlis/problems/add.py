# In this task, each input consists of
# a length-n sequence of depth 2, with all values randomly
# chosen in [0, 1], and the second dimension being all zeros
# except for two elements that are marked by 1. The objective
# is to sum the two random values whose second dimensions are marked by 1.
# We split data in 2 parts, on first part you will train and on second
# part we will test
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

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
