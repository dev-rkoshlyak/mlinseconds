class CaseData:
    def __init__(self, number, limits, train_data, test_data):
        self.number = number
        self.run_seed = number
        self.limits = limits
        self.train_data = train_data
        self.test_data = test_data
        self.description = None

    def set_description(self, description):
        self.description = description
        return self

    def get_limits(self):
        return self.limits
