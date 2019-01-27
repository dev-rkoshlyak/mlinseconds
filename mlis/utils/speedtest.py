import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x

class ConvolModel(nn.Module):
    def __init__(self):
        super(ConvolModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 2)
        self.conv2 = nn.Conv2d(5, 10, 2)
        self.conv3 = nn.Conv2d(10, 10, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.max_pool2d(self.conv3(x), 2)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(x)
        return x

class SpeedCalculator:
    def calc_linear_time_mult(self, use_gpu = False):
        batch_size = 256
        number_of_batches = 100
        expected_steps_per_second = 1000.0
        if use_gpu:
            batch_size *= 10
            expected_steps_per_second /= 10
            number_of_batches *= 10
        input_size = 10
        output_size = 10
        hidden_size = 100
        data = torch.FloatTensor(batch_size*number_of_batches, input_size)
        target = torch.FloatTensor(batch_size*number_of_batches, output_size)
        model = LinearModel(input_size, output_size, hidden_size)
        if use_gpu:
            model = model.cuda()
            data = data.cuda()
            target = target.cuda()
        torch.manual_seed(1)
        data.uniform_(-1.0, 1.0)
        target.uniform_(-1.0, 1.0)

        optimizer = optim.SGD(model.parameters(), lr=0.00001)
        start_time = time.time()
        for ind in range(number_of_batches):
            data_batch = data[batch_size*ind:batch_size*(ind+1)]
            target_batch = target[batch_size*ind:batch_size*(ind+1)]
            optimizer.zero_grad()
            output = model(data_batch)
            loss = F.mse_loss(output, target_batch)
            loss.backward()
            # We don't do step for stability
            # optimizer.step()
        end_time = time.time()
        steps_per_second = number_of_batches/(end_time-start_time)
        return steps_per_second/expected_steps_per_second

    def calc_convol_time_mult(self, use_gpu = False):
        batch_size = 256
        number_of_batches = 100
        expected_steps_per_second = 48.0
        if use_gpu:
            batch_size *= 10
            expected_steps_per_second /= 10
            number_of_batches *= 10
        input_size = 10
        output_size = 40
        hidden_size = 100
        data = torch.FloatTensor(batch_size*number_of_batches, 1, 28, 28)
        target = torch.FloatTensor(batch_size*number_of_batches, output_size)
        model = ConvolModel()
        if use_gpu:
            model = model.cuda()
            data = data.cuda()
            target = target.cuda()
        torch.manual_seed(1)
        data.uniform_(-1.0, 1.0)
        target.uniform_(-1.0, 1.0)

        optimizer = optim.SGD(model.parameters(), lr=0.00001)
        start_time = time.time()
        for ind in range(number_of_batches):
            data_batch = data[batch_size*ind:batch_size*(ind+1)]
            target_batch = target[batch_size*ind:batch_size*(ind+1)]
            optimizer.zero_grad()
            output = model(data_batch)
            loss = F.mse_loss(output, target_batch)
            loss.backward()
            # We don't do step for stability
            # optimizer.step()
        end_time = time.time()
        steps_per_second = number_of_batches/(end_time-start_time)
        return steps_per_second/expected_steps_per_second

class SpeedTest():
    def __init__(self):
        self = self

    def print_speed_report(self):
        speed_calculator = SpeedCalculator()
        linear_time_mult = speed_calculator.calc_linear_time_mult()
        print("Linear CPU time mult = {:.2f}".format(linear_time_mult))
        convol_time_mult = speed_calculator.calc_convol_time_mult()
        print("Convol CPU time mult = {:.2f}".format(convol_time_mult))
        if torch.cuda.is_available():
            linear_time_mult = speed_calculator.calc_linear_time_mult(True)
            print("Linear GPU time mult = {:.2f}".format(linear_time_mult))
            convol_time_mult = speed_calculator.calc_convol_time_mult(True)
            print("Convol GPU time mult = {:.2f}".format(convol_time_mult))
        else:
            print("No cuda")

if __name__ == '__main__':
    SpeedTest().print_speed_report()

