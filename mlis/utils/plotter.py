import pandas as pd
import seaborn as sns

class Filter():
    def __init__(self, columns, filt):
        self.filt = self.build_filter(columns, filt)

    def build_filter(self, columns, filt):
        result = []
        if filt is not None:
            column_to_id = {column: ind for ind, column in enumerate(columns)}
            for key, value in filt.items():
                if type(key) is str:
                    key = column_to_id[key]
                if type(value) not in [tuple, list]:
                    value = [value]
                result.append((key, value))
        return result

    def is_good(self, value, value_list):
        if type(value_list) is tuple:
            return value_list[0] <= value and value <= value_list[1]
        return value in value_list

    def __call__(self, row):
        for key, value in self.filt:
            if not self.is_good(row[key], value):
                return False
        return True

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

    def get_dataframe(self, filt):
        columns_grid = self.get_columns()
        value_types = self.get_value_types()
        columns = columns_grid + ['type', 'value']

        data = []
        for key, values in self.results_data.results_values.items():
            choices = self.results_data.results_choices[key]
            datum = [choices[column] for column in columns_grid]
            if filt(datum):
                for value_type in value_types:
                    for _, value in values[value_type].items():
                        datum_all = datum.copy()
                        datum_all.append(value_type)
                        datum_all.append(value)
                        data.append(datum_all)

        df = pd.DataFrame(data, columns=columns)
        return df

    def show_1d(self, column_index = -1, filt = None):
        columns = self.get_columns()
        filt = Filter(columns, filt)
        x_column = columns.pop(column_index)
        df = self.get_dataframe(filt)
        col_column = Plotter.COLUMN_DELIMITER.join(columns)
        df[col_column] = df.apply (lambda row: Plotter.COLUMN_DELIMITER.join([str(row[c]) for c in columns]), axis=1)
        sns.set(style="darkgrid")
        sns.relplot(x=x_column, y="value", hue="type", col=col_column, col_wrap=2, kind="line", markers=True, data=df)


