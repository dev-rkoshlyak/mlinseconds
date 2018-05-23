import random
from tensorboardX import SummaryWriter

class GridSearch():
    GRID_LIST_SUFFIX = '_grid'
    def __init__(self, solution, randomSearch = True):
        self.solution = solution
        self.solution.__grid_search__ = self
        self.randomSearch = randomSearch
        self.writer = None
        self.enabled = True

    def get_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter()
        return self.writer

    def get_grid_attributes(self, solution):
        s = solution
        return {self.get_grid_attribute(s, a): self.get_grid_attribute_list(s, a) for a in dir(s) if self.filter_grid_attribute(s, a)}

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
        for attr, attr_value in grid_choice.items():
            if len(grid_str):
                grid_str += '|'
            grid_str += attr + ':' + str(attr_value)
        return grid_str

    def get_grid_choice(self, grid_attributes, grid_choice_history):
        history_size = len(grid_choice_history)
        while True:
            grid_choice = {}
            choice_ind = history_size
            for attr, attr_list in grid_attributes.items():
                attr_list_size = len(attr_list)
                if self.randomSearch:
                    attr_ind = random.randint(0, attr_list_size-1)
                else:
                    attr_ind = choice_ind%attr_list_size
                    choice_ind /= attr_list_size
                grid_choice[attr] = attr_list[attr_ind]
            choice_str = self.grid_choice_to_str(grid_choice)
            if choice_str not in grid_choice_history:
                return choice_str, grid_choice

    def set_grid_choice(self, choice_str, grid_choice):
        self.choice_str = choice_str
        for attr, attr_value in grid_choice.items():
            setattr(self.solution, attr, attr_value)

    def search_model(self, case_data, solution, solution_manager):
        if self.enabled == False:
            return
        grid_attributes = self.get_grid_attributes(self.solution)
        grid_size = self.calc_grid_size(grid_attributes)
        if grid_size == 1:
            self.enabled = False
            return
        grid_choice_history = {}
        while len(grid_choice_history) <  grid_size:
            choice_str, grid_choice = self.get_grid_choice(grid_attributes, grid_choice_history)
            self.set_grid_choice(choice_str, grid_choice)
            solution_manager.train_model(solution, case_data)
            grid_choice_history[choice_str] = True
        print(solution_manager.accepted_string("[SEARCH COMPLETED]"))
        print("Specify case_number, if you want to search over other case data")
        exit(0)

    def log_step_value(self, name, value, step):
        if self.enabled == False:
            return
        self.get_writer().add_scalars(name, {self.choice_str: value}, step)

    @classmethod
    def run_case(self, case_data, solution, solution_manager):
        if '__grid_search__' in dir(solution) is not None:
            solution.__grid_search__.search_model(case_data, solution, solution_manager)
