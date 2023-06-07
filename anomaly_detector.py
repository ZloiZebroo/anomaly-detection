import logging
from tools import read_file, flatten, prepare_data, read_models
from db import query_to_df, overwrite_db_table
import argparse
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pyod.models.suod import SUOD as ENSEMBLE

parser = argparse.ArgumentParser(description="Anomaly detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--sql", help="Source location of sql script to load to detector", required=True)
parser.add_argument("--config", help="Path to models configuration file", required=True)
parser.add_argument("--table_to", help="Destination table", required=True)
parser.add_argument("--date_columns", help="Columns with date data type", required=True)
parser.add_argument("--column_by", help="Column by to save result", required=True)
parser.add_argument("--mode", help="Mode", choices=['overwrite', 'increment'], default='overwrite')
parser.add_argument("--over_by", help="Which column over by", required=False)

args = parser.parse_args()
config = vars(args)
logger = logging.getLogger(__name__)
login = {}


# initialization
sql_path = config.get('sql')
models_config = config.get('config')
table_to = config.get('table_to')
over_by = config.get('over_by')
mode = config.get('mode')
column_by = config.get('column_by').split(', ')
date_columns = config.get('date_columns').split(', ')

# log
print(f'mode: {mode}')
print(f'models config path: {models_config}')
print(f'sql_path: {sql_path}')
print(f'date_columns: {date_columns}')
print(f'table_to: {table_to}')
print(f'over_by: {over_by}')
print(f'columns_by: {column_by}')

def main():

    # read query
    query = read_file(sql_path, encoding='utf-8')

    # read models
    models_list = read_models(models_config)

    # det data
    df = query_to_df(query, login=login)

    # iteration over data
    results_list = list()
    over_values = df[over_by].dropna().unique() if over_by else [None]
    for value in over_values:

        # log current value
        print(f'Iterating over: {value}')

        # data
        over_df = df[df[over_by] == value] if over_by else df

        # check data
        data_len = len(over_df)
        if data_len < 2:
            print(f'Too few data: {data_len} rows')
            continue

        # prepare data
        drop_columns = [c for c in column_by if c not in date_columns]
        df_prepared = prepare_data(over_df.drop(columns=drop_columns), date_columns=date_columns)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(df_prepared.values)

        # fit models and get results
        clf = ENSEMBLE(base_estimators=models_list, n_jobs=2, combination='average', verbose=False)
        clf.fit(x)
        pred = clf.predict(x)
        scores = flatten(
            scaler.fit_transform(
                clf.decision_function(x).reshape(-1, 1)
            )
        )

        # get data primary keys
        over_iteration_res_df = over_df[column_by].copy().reset_index(drop=True)

        # join results
        over_iteration_res_df = over_iteration_res_df.join(
            pd.DataFrame({
                'anomaly': pred,
                'score': scores
            })
        )
        results_list.append(over_iteration_res_df)

    # concat results
    result_df = pd.concat(results_list)

    # log result table
    print(result_df)

    # save data to db
    overwrite_db_table(result_df, table_to, login=login)

    return 0

if __name__ == '__main__':
    main()
