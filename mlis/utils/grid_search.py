import random
import pickle
import os
import sys
import torch
import inspect
import collections
import pandas as pd
from ..core.timer import Timer
from ..core.training_context import TrainingContext
from ..core.solution_tester import SolutionTester
from ..core.speed_calculator import SpeedCalculator

class RunsLogs():
    def __init__(self):
        self.run_params_keys = []
        self.clear_data()

    def clear_data(self):
        self.run_params_to_scalars = {}

    def set_run_params_keys(self, run_params_keys):
        if sorted(self.run_params_keys) != sorted(run_params_keys):
            print("[WARNING] run_params keys changed, clearning data")
            self.clear_data()
        self.run_params_keys = run_params_keys

    def get_next_run_seed(self, run_params):
        next_run_seed = 0
        for name, values in self.run_params_to_scalars.get(run_params, {}).items():
            next_run_seed = max(next_run_seed, max(values.keys(), default=-1)+1)
        return next_run_seed

    def log_scalar(self, run_params, name, run_seed, value):
        self.run_params_to_scalars.setdefault(run_params, {}).setdefault(name, {})[run_seed] = value

    def get_scalars(self, run_params, name):
        return list(self.run_params_to_scalars[run_params][name].values())

    def save(self, file_name):
        file_pi = open(file_name, 'wb')
        pickle.dump(self, file_pi)

    def get_value_types(self):
        columns = []
        columns_dict = {}
        for key, value in self.run_params_to_scalars.items():
            for column, _ in value.items():
                if column not in columns_dict:
                    columns.append(column)
                    columns_dict[column] = True
        return columns

    def get_dataframe(self):
        data = []
        for run_params, name_to_scalars in self.run_params_to_scalars.items():
            datum = [run_params.params[run_param_key] for run_param_key in self.run_params_keys]
            for name, scalars in name_to_scalars.items():
                for value in scalars.values():
                    datum_all = datum.copy()
                    datum_all.append(name)
                    datum_all.append(value)
                    data.append(datum_all)

        columns = self.run_params_keys + ['name', 'value']
        df = pd.DataFrame(data, columns=columns)
        return df

    @staticmethod
    def load(file_name):
        if os.path.isfile(file_name):
            return pickle.load(open(file_name, 'rb'))
        print("[WARNING] file with data not exists, return empty results data")
        return RunsLogs()

    @staticmethod
    def get_global():
        global RESULTS_DATA_INSTANCE
        if 'RESULTS_DATA_INSTANCE' not in globals():
            RESULTS_DATA_INSTANCE = RunsLogs()
        return RESULTS_DATA_INSTANCE


class GridSearchConfig():
    RUN_COUNT = 'runs_per_params'
    def __init__(self):
        self.case_number = 1
        self.random_order = False
        self.verbose = False

    def set_case_number(self, case_number):
        self.case_number = case_number
        return self

    def set_random_order(self, random_order):
        self.random_order = random_order
        return self

    def set_verbose(self, verbose):
        self.verbose = verbose
        return self

    def set_runs_params_grid_from_command_line(self, parser):
        all_args = ';'.join(sys.argv)
        runs_per_params = 1
        args = parser.parse_args()
        runs_params_grid = {}
        sorted_attributes = []
        for key, value in vars(args).items():
            if key == self.__class__.RUN_COUNT:
                runs_per_params = value
            else:
                index = all_args.find(key)
                if index == -1:
                    runs_params_grid[key] = value
                else:
                    sorted_attributes.append((index, key, value))
        for _, key, value in sorted(sorted_attributes):
            runs_params_grid[key] = value

        self.set_runs_params_grid(runs_per_params, runs_params_grid)

    def set_runs_params_grid_from_solution(self, solution):
        runs_per_params = self.get_runs_per_params_from_solution(solution)
        s = solution
        code = inspect.getsource(type(s))
        grid_attrs =[(code.index(a), a) for a in dir(s) if self.filter_grid_attribute_key(s, a)]
        grid_attrs = sorted(grid_attrs)

        runs_params_grid = collections.OrderedDict()
        for _, a in grid_attrs:
            runs_params_grid[self.calc_grid_attribute_key(s, a)] = self.get_grid_attribute_list_from_solution(s, a)

        self.set_runs_params_grid(runs_per_params, runs_params_grid)

    def set_runs_config(self, runs_params_grid, runs_per_params):
        self.runs_params_grid = runs_params_grid
        self.runs_per_params = runs_per_params

    def get_runs_per_params_from_solution(self, solution):
        if hasattr(solution, self.__class__.RUN_COUNT):
            return solution.runs_per_params
        else:
            return 1

    def calc_grid_attribute_key(self, obj, attr):
        return attr[:-len(GridSearch.GRID_LIST_SUFFIX)]

    def get_grid_attribute_list_from_solution(self, solution, attribute_key):
        return getattr(solution, attribute_key)

    def filter_grid_attribute_key(self, obj, attr):
        return attr.endswith(GridSearch.GRID_LIST_SUFFIX) and not attr.startswith('__') and not callable(getattr(obj,attr))

class GridSearchContext(TrainingContext):
    def __init__(self, run_seed, timer, case_data, run_idx_per_params, runs_per_params, run_params, runs_logs):
        super(GridSearchContext, self).__init__(run_seed, timer)
        self.type = self.__class__.GRID_SEARCH
        self.case_data = case_data
        self.run_params = run_params
        self.runs_per_params = runs_per_params
        self.run_idx_per_params = run_idx_per_params
        self.runs_logs = runs_logs

    def log_scalar(self, name, value):
        self.runs_logs.log_scalar(self.run_params, name, self.run_seed, value)

    def get_scalars(self, name):
        return self.runs_logs.get_scalars(self.run_params, name)

    def get_scalars_stats(self, name):
        results = self.get_scalars(name)
        t = torch.FloatTensor(results)
        return dict(mean=t.mean().item(), std=t.std().item())

    def __str__(self):
        main_str = self.__class__.__name__ + '(\n'
        for key, value in vars(self).items():
            main_str += "  {}: {}\n".format(key, value)
        main_str += ')'
        return main_str

class RunParams():
    SHORT_PARAM_SEPARATOR = ' '
    SHORT_VALUE_SEPARATOR = ':'

    def __init__(self, params):
        self.params = params
        self.__key_cache__ = self.to_string(sorted_keys=True)

    def __getattr__(self, name):
        return self.params[name]

    def __getstate__(self):
        return self.params

    def __setstate__(self, params):
        self.__init__(params)

    def __hash__(self):
        return hash(self.__key_cache__)

    def __eq__(self, other):
        return other.__key_cache__ == self.__key_cache__

    def __str__(self):
        return self.to_string()

    def to_string(self, sorted_keys=False):
        param_separator = self.__class__.SHORT_PARAM_SEPARATOR
        value_separator = self.__class__.SHORT_VALUE_SEPARATOR
        items = self.params.items()
        items = sorted(items) if sorted_keys else reversed(list(items))
        return param_separator.join(
                [key+value_separator+repr(value) for key, value in items])

class GridSearch():
    GRID_LIST_SUFFIX = '_grid'
    GRID_PARAM_SEPARATOR = ' '
    GRID_VALUE_SEPARATOR = ':'


    def calc_grid_size(self, runs_params_grid):
        grid_size = 1
        for attr, attr_list in runs_params_grid.items():
            grid_size *= len(attr_list)
        return grid_size

    def get_run_params(self, runs_params_grid, grid_runs_history, random_order):
        history_size = len(grid_runs_history)
        while True:
            grid_choice = {}
            choice_ind = history_size
            for attr, attr_list in reversed(list(runs_params_grid.items())):
                attr_list_size = len(attr_list)
                if random_order:
                    attr_ind = random.randint(0, attr_list_size-1)
                else:
                    attr_ind = choice_ind%attr_list_size
                    choice_ind //= attr_list_size
                grid_choice[attr] = attr_list[attr_ind]
            run_params = RunParams(grid_choice)
            if run_params not in grid_runs_history:
                return run_params

    def check_runs_params_grid(self, runs_params_grid):
        for key, value in runs_params_grid.items():
            unique_len = len(set([str(x) for x in value]))
            if len(value) != unique_len:
                raise ValueError('Non unique attributes: {} = {}'.format(key, value))

    def run(self, tester_config, config, runs_logs):
        speed_calculator = SpeedCalculator()
        time_mult = speed_calculator.calc_linear_time_mult()
        print("Local CPU time mult = {:.2f}".format(time_mult))

        self.check_runs_params_grid(config.runs_params_grid)
        runs_logs.set_run_params_keys(list(config.runs_params_grid.keys()))
        grid_size = self.calc_grid_size(config.runs_params_grid)
        if config.verbose:
            print('[Grid search] Runing: grid_size={} runs_per_params={} verbose={}'.format(grid_size, config.runs_per_params, verbose))
        grid_runs_history = {}
        solution_tester = SolutionTester()
        data_provider = tester_config.get_data_provider()
        solution = tester_config.get_solution()
        while len(grid_runs_history) <  grid_size:
            run_params = self.get_run_params(config.runs_params_grid, grid_runs_history, config.random_order)
            case_data = data_provider.create_case_data(config.case_number)
            for run_idx_per_params in range(config.runs_per_params):
                if config.verbose:
                    print('[Grid search] Running: run_params={} run_idx_per_params={}'.format(run_params, run_idx_per_params))
                self.run_seed = runs_logs.get_next_run_seed(run_params)
                limits = case_data.get_limits()
                timer = Timer(limits.time_limit, time_mult)
                context = GridSearchContext(
                        self.run_seed, timer, case_data, run_idx_per_params, config.runs_per_params, run_params, runs_logs)
                solution_tester.train_model(solution, case_data.train_data, context)

            grid_runs_history[run_params] = True
        print(solution_tester.accepted_string("[SEARCH COMPLETED]"))
        return runs_logs
