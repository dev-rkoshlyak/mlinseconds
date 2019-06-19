import random
import torch
import inspect
import collections
from . import solutionmanager as sm
from . import speedtest

class ResultsData():
    @staticmethod
    def get_global():
        global RESULTS_DATA_INSTANCE
        if 'RESULTS_DATA_INSTANCE' not in globals():
            RESULTS_DATA_INSTANCE = ResultsData()
        return RESULTS_DATA_INSTANCE

    def __init__(self):
        self.clear_data()

    def clear_data(self):
        self.grid_attributes = None
        self.results_choices = {}
        self.results_values = {}

    def set_grid_attributes(self, grid_attributes):
        if self.grid_attributes is not None and sorted(list(self.grid_attributes.keys())) != sorted(list(grid_attributes.keys())):
            print("[WARNING] grid attributes changed, clearning data")
            self.clear_data()
        self.grid_attributes = grid_attributes

    def get_next_init_seed(self, choice_str):
        next_init_seed = 0
        for name, values in self.results_values.get(choice_str, {}).items():
            next_init_seed = max(next_init_seed, max(values.keys(), default=-1)+1)
        return next_init_seed

    def add_result(self, choice_str, grid_choice, name, init_seed, value):
        if choice_str not in self.results_values:
            self.results_choices[choice_str] = grid_choice
            self.results_values[choice_str] = {}
        if name not in self.results_values[choice_str]:
            self.results_values[choice_str][name] = {}
        self.results_values[choice_str][name][init_seed] = value

    def get_results(self, choice_str, name):
        return list(self.results_values[choice_str][name].values())

class GridSearch():
    GRID_LIST_SUFFIX = '_grid'
    GRID_PARAM_SEPARATOR = ' '
    GRID_VALUE_SEPARATOR = ':'

    def __init__(self):
        self.results_data = None

    def get_grid_attributes(self, solution):
        s = solution
        code = inspect.getsource(type(s))
        grid_attrs =[(code.index(a), a) for a in dir(s) if self.filter_grid_attribute(s, a)]
        grid_attrs = sorted(grid_attrs)

        grid_attributes = collections.OrderedDict()
        for _, a in grid_attrs:
            grid_attributes[self.get_grid_attribute(s, a)] = self.get_grid_attribute_list(s, a)
        return grid_attributes

    def get_grid_attribute(self, obj, attr):
        return attr[:-len(GridSearch.GRID_LIST_SUFFIX)]

    def get_grid_attribute_list(self, obj, attr):
        return getattr(obj, attr)

    def filter_grid_attribute(self, obj, attr):
        return attr.endswith(GridSearch.GRID_LIST_SUFFIX) and not attr.startswith('__') and not callable(getattr(obj,attr))

    def calc_grid_size(self, grid_attributes):
        grid_size = 1
        for attr, attr_list in grid_attributes.items():
            grid_size *= len(attr_list)
        return grid_size

    def grid_choice_to_str(self, grid_choice):
        grid_str = ''
        for attr, attr_value in reversed(grid_choice.items()):
            if len(grid_str):
                grid_str += GridSearch.GRID_PARAM_SEPARATOR
            grid_str += attr + GridSearch.GRID_VALUE_SEPARATOR + str(attr_value)
        return grid_str

    def get_grid_choice(self, grid_attributes, grid_choice_history, random_order):
        history_size = len(grid_choice_history)
        while True:
            grid_choice = collections.OrderedDict()
            choice_ind = history_size
            for attr, attr_list in reversed(grid_attributes.items()):
                attr_list_size = len(attr_list)
                if random_order:
                    attr_ind = random.randint(0, attr_list_size-1)
                else:
                    attr_ind = choice_ind%attr_list_size
                    choice_ind //= attr_list_size
                grid_choice[attr] = attr_list[attr_ind]
            choice_str = self.grid_choice_to_str(grid_choice)
            if choice_str not in grid_choice_history:
                return choice_str, grid_choice

    def set_grid_choice(self, solution, choice_str, grid_choice):
        self.choice_str = choice_str
        self.grid_choice = grid_choice
        for attr, attr_value in grid_choice.items():
            setattr(solution, attr, attr_value)

    def add_result(self, name, value):
        self.results_data.add_result(self.choice_str, self.grid_choice, name, self.init_seed, value)

    def get_results(self, name):
        return self.results_data.get_results(self.choice_str, name)

    def get_stats(self, name):
        results = self.get_results(name)
        t = torch.FloatTensor(results)
        return t.mean().item(), t.std().item()

    def get_iter_number(self, solution):
        if hasattr(solution, 'iter_number'):
            return solution.iter_number
        else:
            return 1

    def run(self, config, case_number, random_order=False, verbose=False, results_data = None):
        speed_calculator = speedtest.SpeedCalculator()
        time_mult = speed_calculator.calc_linear_time_mult()
        print("Local CPU time mult = {:.2f}".format(time_mult))
        solution = config.get_solution()
        solution.grid_search = self
        if results_data is None:
            results_data = ResultsData()
        self.results_data = results_data
        grid_attributes = self.get_grid_attributes(solution)
        self.results_data.set_grid_attributes(grid_attributes)
        grid_size = self.calc_grid_size(grid_attributes)
        iter_number = self.get_iter_number(solution)
        if verbose:
            print('[Grid search] Runing: grid_size={} iter_number={} verbose={}'.format(grid_size, iter_number, verbose))
        grid_choice_history = {}
        solution_manager = sm.SolutionManager()
        data_provider = config.get_data_provider()
        while len(grid_choice_history) <  grid_size:
            choice_str, grid_choice = self.get_grid_choice(grid_attributes, grid_choice_history, random_order)
            self.set_grid_choice(solution, choice_str, grid_choice)
            case_data = data_provider.create_case_data(case_number)
            for iter in range(iter_number):
                if verbose:
                    print('[Grid search] Running: choise_str={} iter={}'.format(choice_str, iter))
                solution.iter = iter
                self.init_seed = results_data.get_next_init_seed(choice_str)
                solution_manager.train_model(self.init_seed, solution, case_data)

            grid_choice_history[choice_str] = True
        print(solution_manager.accepted_string("[SEARCH COMPLETED]"))
        print("Specify case_number, if you want to search over other case data")
