import pandas as pd
import seaborn as sns

class Plotter():
    COLUMN_DELIMITER = ':'
    def __init__(self, results_data):
        self.results_data = results_data

    def get_columns(self):
        return list(iter(self.results_data.grid_attributes.keys()))

    def get_value_types(self):
        columns_grid = list(iter(self.results_data.grid_attributes.keys()))
        columns = []
        columns_dict = {}
        for key, value in self.results_data.results_values.items():
            for column, _ in value.items():
                if column not in columns_dict:
                    columns.append(column)
                    columns_dict[column] = True
        return columns

    def get_dataframe(self):
        columns_grid = self.get_columns()
        value_types = self.get_value_types()
        columns = columns_grid + ['type', 'value']

        data = []
        for key, values in self.results_data.results_values.items():
            choices = self.results_data.results_choices[key]
            datum = [choices[column] for column in columns_grid]
            for value_type in value_types:
                for _, value in values[value_type].items():
                    datum_all = datum.copy()
                    datum_all.append(value_type)
                    datum_all.append(value)
                    data.append(datum_all)

        df = pd.DataFrame(data, columns=columns)
        return df

    def show_1d(self, column_index = -1, query = None):
        columns = self.get_columns()
        x_column = columns.pop(column_index)
        df = self.get_dataframe()
        if query is not None:
            df.query(query, inplace=True)
        if df.shape[0] == 0:
            print('[WARNING] Empty filtered results')
            return
        col_column = 'col:'+Plotter.COLUMN_DELIMITER.join(columns)
        df[col_column] = df.apply (lambda row: Plotter.COLUMN_DELIMITER.join([str(row[c]) for c in columns]), axis=1)
        sns.set(style="darkgrid")
        sns.relplot(x=x_column, y="value", hue="type", col=col_column, col_wrap=2, kind="line", markers=True, data=df)


