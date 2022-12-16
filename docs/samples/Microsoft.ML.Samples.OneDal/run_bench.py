import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import subprocess
import sys
import os
import io


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def convert_to_csv(data, filename, label='target'):
    data = pd.DataFrame(data, columns=[f'f{i}' for i in range(data.shape[1] - 1)] + ['target'])
    data[label] = data[label].astype(int)
    data.to_csv(filename, index=False)


def read_output_from_command(command, env):
    res = subprocess.run(command.split(' '), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8', env=env)
    return res.stdout[:-1], res.stderr[:-1]


def run_case(n_samples, n_features, n_trees, n_leaves, n_runs, task_type):
    data_params = {
        'n_samples': 2 * n_samples,
        'n_features': n_features,
        'n_informative': n_features // 2,
        'random_state': RANDOM_STATE
    }
    if task_type == 'binary':
        x, y = make_classification(**data_params, class_sep=0.7)
    elif task_type == 'regression':
        x, y = make_regression(**data_params, noise=0.33)

    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=RANDOM_STATE)

    train_filename, test_filename = "synth_data_train.csv", "synth_data_test.csv"
    convert_to_csv(train_data, train_filename)
    convert_to_csv(test_data, test_filename)

    print(f'n_samples={n_samples}, n_features={n_features}, n_trees={n_trees}, n_leaves={n_leaves}')

    case_dict = {'algorithm': [f'Random Forest {task_type.capitalize()}'],
        'n samples': [n_samples], 'n features': [n_features],
        'n trees': [n_trees], 'n leaves': [n_leaves]}

    if task_type == 'binary':
        metrics = ['accuracy', 'F1 score']
    elif task_type == 'regression':
        metrics = ['RMSE', 'R2 score']
    metrics = ['all workflow time[ms]', f'training {metrics[0]}', f'testing {metrics[0]}', f'training {metrics[1]}', f'testing {metrics[1]}']
    metrics_map_template = {el: '{} ' + f'{el}' for el in metrics}

    result = None
    for i in range(n_runs):
        env_copy = os.environ.copy()
        default_stdout, default_stderr = read_output_from_command(f'dotnet run {train_filename} {test_filename} {task_type} RandomForest {n_trees} {n_leaves}', env_copy)

        print('DEFAULT STDOUT:', default_stdout, 'DEFAULT STDERR:', default_stderr, sep='\n')

        default_res = io.StringIO(default_stdout + '\n')
        default_res = pd.DataFrame(case_dict).merge(pd.read_csv(default_res))

        env_copy['MLNET_BACKEND'] = 'ONEDAL'
        optimized_stdout, optimized_stderr = read_output_from_command(f'dotnet run {train_filename} {test_filename} {task_type} RandomForest {n_trees} {n_leaves}', env_copy)

        print('OPTIMIZED STDOUT:', optimized_stdout, 'OPTIMIZED STDERR:', optimized_stderr, sep='\n')

        optimized_res = io.StringIO(optimized_stdout + '\n')
        optimized_res = pd.read_csv(optimized_res)

        default_res = default_res.rename(columns={k: v.format('ML.NET') for k, v in metrics_map_template.items()})
        optimized_res = optimized_res.rename(columns={k: v.format('oneDAL') for k, v in metrics_map_template.items()})

        if result is None:
            result = default_res.merge(optimized_res)
        else:
            result = pd.concat([result, default_res.merge(optimized_res)], axis=0)

    all_columns = list(result.columns)
    groupby_columns = list(case_dict.keys())
    metric_columns = list(result.columns)
    for column in groupby_columns:
        metric_columns.remove(column)
    result = result.groupby(groupby_columns)[metric_columns].mean().reset_index()

    return result

n_samples_range = [5000, 10000]
n_features_range = [8, 64]
n_trees_range = [100]
n_leaves_range = [128]
n_runs = 5

result = None
for task_type in ['binary', 'regression']:
    for n_samples in n_samples_range:
        for n_features in n_features_range:
            for n_trees in n_trees_range:
                for n_leaves in n_leaves_range:
                    new_result = run_case(n_samples, n_features, n_trees, n_leaves, n_runs, task_type=task_type)
                    if result is None:
                        result = new_result
                    else:
                        result = pd.concat([result, new_result], axis=0)

result.to_csv('result.csv', index=False)
result.to_csv(sys.stdout, index=False)
