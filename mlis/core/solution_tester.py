import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .speed_calculator import SpeedCalculator
from .training_context import TrainingContext
from .timer import Timer

class SolutionTester():
    HINT_YELLOW = '\033[93m'
    ACCEPTED_GREEN = '\033[92m'
    REJECTED_RED = '\033[91m'
    END_COLOR = '\033[0m'
    def __init__(self):
        self = self

    def calc_model_size(self, model):
        modelSize = 0
        for param in model.parameters():
            modelSize += param.view(-1).size()[0]
        return modelSize

    def sampleData(self, data, max_samples = 1000):
        dataSize = list(data.size())
        data = data.view(dataSize[0], -1)[:max_samples,:]
        dataSize[0] = min(dataSize[0], max_samples)
        data = data.view(tuple(dataSize))
        return data

    def calc_model_stats(self, config, model, data, target):
        with torch.no_grad():
            data = self.sampleData(data)
            target = self.sampleData(target)
            output = model(data)
            # Number of correct predictions
            predict = model.calc_predict(output)
            error = model.calc_error(output, target)
            correct = predict.eq(target.view_as(predict)).long().sum()
            total = predict.view(-1).size(0)
            return {
                    'error': error.item(),
                    'correct': correct.item(),
                    'total': total,
                    'accuracy': correct.item()/total,
                    }

    def train_model(self, solution, train_data, context):
        # We need to init random system, used for multiple runs
        torch.manual_seed(context.run_seed)
        model = solution.train_model(*train_data, context)
        execution_time = context.timer.get_execution_time()
        reject_reason = context.get_reject_reason()
        return context.step, execution_time, reject_reason, model

    def run_case(self, config, case_data):
        solution = config.get_solution()
        run_seed = case_data.number
        context = None
        step, execution_time, reject_reason, model = self.train_model(solution, context)
        model_size = self.calc_model_size(model)
        model.eval()
        data, target = case_data.train_data
        train_stat = self.calc_model_stats(config, model, data, target)
        data, target = case_data.test_data
        test_stat = self.calc_model_stats(config, model, data, target)
        return {
                'case': case_data.number,
                'step': step,
                'time': execution_time,
                'reject_reason': reject_reason,
                'size': model_size,
                'trainStat': train_stat,
                'testStat': test_stat
                }

    def run_case_from_test_config(self, config, test_config):
        data_provider = config.get_data_provider()
        solution = config.get_solution()
        case_data = data_provider.create_case_data(test_config)
        limits = case_data.get_limits()
        time_mult = 1.0
        timer = Timer(limits.time_limit, time_mult)
        context = TrainingContext(case_data.run_seed, timer)
        step, execution_time, reject_reason, model = self.train_model(solution, case_data.train_data, context)
        model_size = self.calc_model_size(model)
        model.eval()
        data, target = case_data.train_data
        train_stat = self.calc_model_stats(config, model, data, target)
        data, target = case_data.test_data
        test_stat = self.calc_model_stats(config, model, data, target)
        return {
            'modelSize': model_size,
            'trainingSteps': step,
            'trainingTime': execution_time,
            'evaluationTime': 3.0,
            'trainError': train_stat['error'],
            'trainCorrect': train_stat['correct'],
            'trainTotal': train_stat['total'],
            'trainAccuracy': train_stat['accuracy'],
            'trainMetric': 3.0,
            'testError': test_stat['error'],
            'testCorrect': test_stat['correct'],
            'testTotal': test_stat['total'],
            'testAccuracy': test_stat['accuracy'],
            'testMetric': 3.0,
        }

    @classmethod
    def colored_string(self, s, color):
        return color+s+SolutionTester.END_COLOR

    @classmethod
    def print_hint(self, s, step=0):
        if step==0:
            print(SolutionTester.colored_string(s, SolutionTester.HINT_YELLOW))


    def hint_string(self, s):
        return self.colored_string(s, SolutionTester.HINT_YELLOW)

    def rejected_string(self, s):
        return self.colored_string(s, SolutionTester.REJECTED_RED)

    def accepted_string(self, s):
        return self.colored_string(s, SolutionTester.ACCEPTED_GREEN)

    def evaluate_result(self, case_data, case_result):
        limits = case_data.get_limits()
        r = case_result
        case = r['case']
        description = r['description']
        step = r['step']
        size = r['size']
        time = r['time']
        reject_reason = r['reject_reason']
        train_error = r['trainStat']['error']
        train_correct = r['trainStat']['correct']
        train_total = r['trainStat']['total']
        train_ration = train_correct/float(train_total)
        test_error = r['testStat']['error']
        test_correct = r['testStat']['correct']
        test_total = r['testStat']['total']
        test_ratio = test_correct/float(test_total)

        print("Case #{}[{}] Step={} Size={}/{} Time={:.1f}/{:.1f}".format(
            case, description, step, size, limits.size_limit, time, limits.time_limit))
        print("Train correct/total={}/{} Ratio/limit={:.2f}/{:.2f} Error={}".format(
            train_correct, train_total, train_ration, limits.test_limit, train_error))
        print("Test  correct/total={}/{} Ratio/limit={:.2f}/{:.2f} Error={}".format(
            test_correct, test_total, test_ratio, limits.test_limit, test_error))
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

    def run(self, config, case_number):
        speed_calculator = SpeedCalculator()
        time_mult = speed_calculator.calc_linear_time_mult()
        print("Local CPU time mult = {:.2f}".format(time_mult))
        data_provider = config.get_data_provider()
        if case_number == -1:
            casses = [i+1 for i in range(data_provider.number_of_cases)]
        else:
            casses = [case_number]
        case_results = []
        case_limits = []
        for case in casses:
            case_data = data_provider.create_case_data(case)
            case_result = self.run_case(config, case_data)
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
        if case_number == -1:
            print(self.accepted_string("[ACCEPTED]")+" you can submit now your score")
            print("In order to submit just do a Facebook comment with your score")
        else:
            print(self.hint_string("[GOOD]")+" test passed, try to run on all tests")

