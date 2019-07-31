from ..core.solution_tester import SolutionTester
from .hello_xor_problem import DataProvider
from .hello_xor_solution import Solution

class TesterConfig:
    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

def main():
    SolutionTester().run(TesterConfig(), case_number=-1)

if __name__ == '__main__"':
    main()
