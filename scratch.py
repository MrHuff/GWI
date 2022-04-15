import dill
import pickle

import pandas as pd
import torch
from hyperopt import Trials, STATUS_OK
import itertools


def extract_regression_data(results_folder):
    dataset = ['boston', 'concrete', 'energy','KIN8NM', 'power','protein' ,'wine', 'yacht', 'naval']
    folds=[0,1,2,3,4,5,6,7,8,9]
    p_list = list(itertools.product(dataset,folds))
    data_list =[]
    for (ds,f) in p_list:
        load_data_base_string = f'{results_folder}/{ds}_seed=0_fold_idx={f}_model=GWI/hyperopt_database.p'
        try:
            trials = pickle.load(open(load_data_base_string, "rb"))
            trials = dill.loads(trials)
            best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=False)[0]
            data_list.append([ds,f,best_trial['loss'],best_trial['test_loss'],best_trial['val_r2'],best_trial['test_r2'],
                              best_trial['test_rsme'],best_trial['T']])
        except Exception as e:
            print('file missing, job probably did not run properly')
    all_jobs = pd.DataFrame(data_list,columns=['dataset','fold','val_acc','NLL','val_r2','test_r2','rsme','T'])
    all_jobs=all_jobs.sort_values(['dataset','fold'])
    all_jobs.to_csv(f'{results_folder}/all_jobs.csv')
    summary_df_r2 = all_jobs.groupby(['dataset'])['test_r2'].mean()
    summary_df_std_r2 = all_jobs.groupby(['dataset'])['test_r2'].std()
    new_latex_df=summary_df_r2.apply(lambda x: rf'${round(x,3)} \pm')+summary_df_std_r2.apply(lambda x: f'{round(x,3)}$')
    new_latex_df.to_latex(buf=f"{results_folder}/final_results_latex_r2.tex",escape=False)
    summary_df = all_jobs.groupby(['dataset'])['NLL'].mean()
    summary_df_std = all_jobs.groupby(['dataset'])['NLL'].std()
    new_latex_df=summary_df.apply(lambda x: rf'${round(x,3)} \pm')+summary_df_std.apply(lambda x: f'{round(x,3)}$')
    new_latex_df.to_latex(buf=f"{results_folder}/final_results_latex.tex",escape=False)

if __name__ == '__main__':
    # extract_regression_data('regression_test_True')
    # extract_regression_data('regression_test_False')
    extract_regression_data('regression_test_1_False')
    extract_regression_data('regression_test_1_True')
    # extract_regression_data('regression_test_15_False')
    # extract_classification_data('learned_z_class_2_results')