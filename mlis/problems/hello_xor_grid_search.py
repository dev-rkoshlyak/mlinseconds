import argparse
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from ..utils.grid_search import RunsLogs, GridSearchConfig, GridSearch
from .hello_xor_tester import TesterConfig

def main():
    parser = argparse.ArgumentParser(description='Grid search', allow_abbrev=False)
    parser.add_argument('-run_count', type=int, default=10)
    parser.add_argument('-hidden_size', type=int, nargs='+', default=[1])
    parser.add_argument('-learning_rate', type=float, nargs='+', default=[1.0])
    tester_config = TesterConfig()
    grid_search_config = GridSearchConfig()
    grid_search_config.set_case_number(1)
    grid_search_config.set_verbose(False)
    #grid_search_config.set_grid_attributes_from_command_line(parser)
    grid_search_config.set_runs_config(
            runs_params_grid = dict(
                hidden_size=[3, 4],
                learning_rate=[1.0,2.0],
                ),
            runs_per_params=10
            )
    #grid_search_config.set_grid_attributes_from_solution(tester_config.get_solution())

    # Note: we can save and accumulate results data if grid keys did not change
    runs_logs_file = 'helloxor_runs_logs.pickle'
    runs_logs = RunsLogs.load(runs_logs_file)
    runs_logs = GridSearch().run(tester_config, grid_search_config, runs_logs)
    runs_logs.save(runs_logs_file)

    # Explore data
    df = runs_logs.get_dataframe()
    def confidence_range(df):
        res = bs.bootstrap(df.values, stat_func=bs_stats.mean)
        return res.lower_bound, res.upper_bound
    df = df.groupby(['name', 'hidden_size', 'learning_rate']).agg({'value':['count', 'min', 'mean', 'max', confidence_range]})
    df.columns = df.columns.map('_'.join)
    df = df.reset_index()
    df['value_lmean'] = df.value_confidence_range.transform(lambda x: x[0])
    df['value_umean'] = df.value_confidence_range.transform(lambda x: x[1])
    df = df.drop(columns=['value_confidence_range'])
    print(df)

if __name__ == '__main__':
    main()
