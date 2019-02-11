# The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
#
# In this problem you need to implement model that will learn to recognize
# handwritten digits
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
        self.output_size = output_size
        sm.SolutionManager.print_hint("Hint[1]: Experement with more deep networks")
        self.hidden_size = 10
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        sm.SolutionManager.print_hint("Hint[2]: You probably would need convolution network for this task")
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def calc_loss(self, output, target):
        loss = F.nll_loss(output, target)
        return loss

    def calc_predict(self, output):
        predict = output.max(1, keepdim=True)[1]
        return predict

class Solution():
    def __init__(self):
        self = self

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            optimizer = optim.SGD(model.parameters(), lr=1.0)
            sm.SolutionManager.print_hint("Hint[3]: Experement with approximating deriviative based on subset of data", step)
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            sm.SolutionManager.print_hint("Hint[4]: Experement with other activation fuctions", step)
            output = model(data)
            # get the index of the max probability
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # calculate loss
            loss = model.calc_loss(output, target)
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
        self.test_limit = 0.95

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0,mask.size(0))[mask].long()
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0,1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
