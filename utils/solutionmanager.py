import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from contextlib import contextmanager

class CpuSpeedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CpuSpeedModel, self).__init__()
        hidden_size = 100
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
        x = self.linear3(x)
        x = F.sigmoid(x)
        return x

class CpuSpeed:
    def calc_time_mult(self):
        batch_size = 256
        number_of_batches = 100
        input_size = 10
        output_size = 10
        model = CpuSpeedModel(input_size, output_size)
        data = torch.FloatTensor(batch_size*number_of_batches, input_size)
        target = torch.FloatTensor(batch_size*number_of_batches, output_size)
        torch.manual_seed(1)
        data.uniform_(-1.0, 1.0)
        target.uniform_(-1.0, 1.0)

        start_time = time.time()
        optimizer = optim.SGD(model.parameters(), lr=0.00001)
        for ind in range(number_of_batches):
            data_batch = data[batch_size*ind:batch_size*(ind+1)]
            target_batch = target[batch_size*ind:batch_size*(ind+1)]
            optimizer.zero_grad()
            output = model(data_batch)
            loss = F.mse_loss(output, target_batch)
            loss.backward()
            # optimizer.step()
        end_time = time.time()
        steps_per_second = number_of_batches/(end_time-start_time)
        return steps_per_second/1200.0

class CaseData:
    def __init__(self, number, limits, train_data, test_data):
        self.number = number
        self.limits = limits
        self.train_data = train_data
        self.test_data = test_data
        self.description = None
        self.input_size = train_data[0][0].view(-1).size()[0]
        self.output_size = 1

    def set_description(self, description):
        self.description = description
        return self

    def set_output_size(self, output_size):
        self.output_size = output_size
        return self

    def get_limits(self):
        return self.limits

class Timer():
    def __init__(self, time_limit, time_mult):
        self.time_limit = time_limit
        self.time_mult = time_mult
        self.start_time = time.time()
        self.pause_time = 0.0

    @contextmanager
    def pause(self):
        pause_start = time.time()
        yield None
        pause_end = time.time()
        self.pause_time += pause_end-pause_start

    def get_time_left(self):
        return self.time_limit - self.get_execution_time()

    def get_execution_time(self):
        return (time.time() - self.start_time - self.pause_time) * self.time_mult

    def get_pause_time(self):
        return self.pause_time


class TrainingContext():
    def __init__(self, case_data, timer):
        self.case_data = case_data
        self.timer = timer
        self.case_data_accessed = False

    def get_case_data(self):
        self.case_data_accessed = True
        return self.case_data

    def get_timer(self):
        return self.timer

    def get_reject_reason(self):
        if self.timer.pause_time > 0.0:
            return "Timer paused"
        if self.case_data_accessed:
            return "Case data accessed"
        return None

class SolutionManager():
    HINT_YELLOW = '\033[93m'
    ACCEPTED_GREEN = '\033[92m'
    REJECTED_RED = '\033[91m'
    END_COLOR = '\033[0m'
    def __init__(self, config):
        self.config = config

    def calc_model_size(self, model):
        modelSize = 0
        for param in model.parameters():
            modelSize += param.view(-1).size()[0]
        return modelSize

    def sampleData(self, data, max_samples):
        dataSize = list(data.size())
        data = data.view(dataSize[0], -1)[:max_samples,:]
        dataSize[0] = min(dataSize[0], max_samples)
        data = data.view(tuple(dataSize))
        return data

    def calc_model_stats(self, model, data, target):
        with torch.no_grad():
            data = self.sampleData(data, self.config.max_samples)
            target = self.sampleData(target, self.config.max_samples)
            output = model(data)
            if output[0].size()[0] == 1:
                predict = output.round()
            else:
                predict = output.max(1, keepdim=True)[1]
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum()
            total = target.view(-1).size(0)
            if output[0].size(0) == 1:
                loss = F.mse_loss(output, target)
            else:
                loss = F.nll_loss(output, target)
            return {
                    'loss': loss.item(),
                    'correct': correct.item(),
                    'total': total
                    }

    def train_model(self, solution, case_data):
        input_size = case_data.input_size
        output_size = case_data.output_size
        limits = case_data.get_limits()
        data, target = case_data.train_data
        cpuSpeed = CpuSpeed()
        time_mult = cpuSpeed.calc_time_mult()
        torch.manual_seed(case_data.number)
        timer = Timer(limits.time_limit, time_mult)
        model = solution.create_model(input_size, output_size)
        context = TrainingContext(case_data, timer)
        step = solution.train_model(model, data, target, context)
        execution_time = timer.get_execution_time()
        reject_reason = context.get_reject_reason()
        return step, execution_time, reject_reason, model

    def run_case(self, case_data):
        solution = self.config.get_solution()
        # grid search integration
        if '__grid_search__' in dir(solution) is not None:
            solution.__grid_search__.search_model(case_data, solution, self)

        step, execution_time, reject_reason, model = self.train_model(solution, case_data)
        model_size = self.calc_model_size(model)
        model.eval()
        data, target = case_data.train_data
        train_stat = self.calc_model_stats(model, data, target)
        data, target = case_data.test_data
        test_stat = self.calc_model_stats(model, data, target)
        return {
                'case': case_data.number,
                'step': step,
                'time': execution_time,
                'reject_reason': reject_reason,
                'size': model_size,
                'trainStat': train_stat,
                'testStat': test_stat
                }

    @classmethod
    def colored_string(self, s, color):
        return color+s+SolutionManager.END_COLOR

    @classmethod
    def print_hint(self, s, step=0):
        if step==0:
            print(SolutionManager.colored_string(s, SolutionManager.HINT_YELLOW))


    def rejected_string(self, s):
        return self.colored_string(s, SolutionManager.REJECTED_RED)

    def accepted_string(self, s):
        return self.colored_string(s, SolutionManager.ACCEPTED_GREEN)

    def evaluate_result(self, case_data, case_result):
        limits = case_data.get_limits()
        r = case_result
        case = r['case']
        description = r['description']
        step = r['step']
        size = r['size']
        time = r['time']
        reject_reason = r['reject_reason']
        train_loss = r['trainStat']['loss']
        train_correct = r['trainStat']['correct']
        train_total = r['trainStat']['total']
        train_ration = train_correct/float(train_total)
        test_loss = r['testStat']['loss']
        test_correct = r['testStat']['correct']
        test_total = r['testStat']['total']
        test_ratio = test_correct/float(test_total)

        print("Case #{}[{}] Step={} Size={}/{} Time={:.1f}/{:.1f}".format(
            case, description, step, size, limits.size_limit, time, limits.time_limit))
        print("Train correct/total={}/{} Ratio/limit={:.2f}/{:.2f} Loss={}".format(
            train_correct, train_total, train_ration, limits.test_limit, train_loss))
        print("Test  correct/total={}/{} Ratio/limit={:.2f}/{:.2f} Loss={}".format(
            test_correct, test_total, test_ratio, limits.test_limit, test_loss))
        r['accepted'] = False
        if size > limits.size_limit:
            print(self.rejected_string("[REJECTED]")+": MODEL IS TOO BIG: Size={} Size Limit={}".format(size, limits.size_limit))
        elif time > limits.time_limit:
            print(self.rejected_string("[REJECTED]")+": TIME LIMIT EXCEEDED: Time={:.1f} Time Limit={:.1f}".format(time, limits.time_limit))
        elif test_ratio < limits.test_limit:
            print(self.rejected_string("[REJECTED]")+": MODEL DID NOT LEARN: Learn ratio={}/{}".format(test_ratio, limits.test_limit))
        elif reject_reason is not None:
            print(self.rejected_string("[REJECTED]")+": " + reject_reason)
        else:
            r['accepted'] = True
            print(self.accepted_string("[OK]"))

        return r

    def run(self, case_number):
        cpuSpeed = CpuSpeed()
        time_mult = cpuSpeed.calc_time_mult()
        print("Local CPU time mult = {:.2f}".format(time_mult))
        data_provider = self.config.get_data_provider()
        if case_number == -1:
            casses = [i+1 for i in range(data_provider.number_of_cases)]
        else:
            casses = [case_number]
        case_results = []
        case_limits = []
        for case in casses:
            case_data = data_provider.create_case_data(case)
            case_result = self.run_case(case_data)
            case_result['description'] = case_data.description
            case_result = self.evaluate_result(case_data, case_result)
            if case_result['accepted'] == False:
                print("Need more hint??? Ask for hint at Facebook comments")
                return False
            case_limits.append(case_data.get_limits())
            case_results.append(case_result)

        test_rates = [x['testStat']['correct']/float(x['testStat']['total']) for x in case_results]
        test_rate_max = max(test_rates)
        test_rate_mean = sum(test_rates)/len(test_rates)
        test_rate_min = min(test_rates)
        num_cases = float(len(case_results))
        step_mean = sum([x['step'] for x in case_results])/num_cases
        time_mean = sum([x['time'] for x in case_results])/num_cases
        size_mean = sum([x['size'] for x in case_results])/num_cases
        test_limit_mean = sum([x.test_limit for x in case_limits])/num_cases
        time_limit_mean = sum([x.time_limit for x in case_limits])/num_cases
        size_limit_mean = sum([x.size_limit for x in case_limits])/num_cases
        print("Test rate (max/mean/min/limit) = {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(
            test_rate_max, test_rate_mean, test_rate_min, test_limit_mean))
        print("Average steps = {:.3f}".format(step_mean))
        print("Average time = {:.3f}/{:.3f}".format(time_mean, time_limit_mean))
        print("Average size = {:.3f}/{:.3f}".format(size_mean, size_limit_mean))
        print(self.accepted_string("[ACCEPTED]")+" you can submit now your score")
        print("In order to submit just do a Facebook comment with your score")

