from ..core.solution_tester import SolutionTester
from .problem import DataProvider
from .solution import Solution
import json

class TesterConfig:
    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

def main():
    with open('test.config', 'r') as myfile:
        test_config = json.loads(myfile.read())
    test_result = SolutionTester().run_case_from_test_config(TesterConfig(), test_config)
    with open("test.result", "w") as myfile:
        myfile.write(json.dumps(test_result))

if __name__ == '__main__':
    main()
