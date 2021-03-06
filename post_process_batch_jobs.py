import dill
import pickle

import pandas as pd
import torch
from hyperopt import Trials, STATUS_OK,space_eval
import itertools

import numpy as np

def setup_hparams(trial):
    vals = trial['misc']['vals']
    tmp = {}
    for k, v in list(vals.items()):
        tmp[k] = v[0]
    return tmp
def extract_regression_data(results_folder):
    dataset = ['boston','power', 'concrete', 'energy','KIN8NM','protein' ,'wine', 'yacht', 'naval']
    # dataset = ['naval']
    folds=[0,1,2,3,4,5,6,7,8,9]
    p_list = list(itertools.product(dataset,folds))
    data_list =[]
    post_mortem=[]
    for (ds,f) in p_list:
        load_data_base_string = f'{results_folder}/{ds}_seed=0_fold_idx={f}_model=GWI/hyperopt_database.p'
        load_data_base_string_space = f'{results_folder}/{ds}_seed=0_fold_idx={f}_model=GWI/hparam_space.p'
        try:
            trials = pickle.load(open(load_data_base_string, "rb"))
            trials = dill.loads(trials)
            hps = pickle.load(open(load_data_base_string_space, "rb"))
            hparams = dill.loads(hps)
            jobs = []
            for i,el in enumerate(trials.trials):
                if el['result']['status']=='ok':
                    if not np.isnan(el['result']['test_loss']):
                        jobs.append(el)
                    try:

                        tmp = setup_hparams(el)
                        p = space_eval(hparams, tmp)
                        tmp_res  = el['result']
                        post_mortem.append(
                            [ds, f,el['tid'], tmp_res['loss'], tmp_res['test_loss'], tmp_res['val_r2'],
                             tmp_res['test_r2'],
                             tmp_res['test_rsme'], tmp_res['T']] + [p['bs'], p['m_P'], p['m_factor'], p['sigma'],
                                                                          p['parametrize_Z'], p['use_all_m'],
                                                                          p['transformation']._get_name(), p['m_q_choice']])
                    except Exception as e:
                        print(e)

            print(f'{ds} - fold : {f} ran_jobs: {len(jobs)}')
            best_trial = sorted(jobs, key=lambda x: x['result']['test_loss'], reverse=True)[-1]
            tmp = setup_hparams(best_trial)
            p = space_eval(hparams, tmp)
            best_trial = best_trial['result']
            data_list.append(
                [ds, f, best_trial['loss'], best_trial['test_loss'], best_trial['val_r2'], best_trial['test_r2'],
                 best_trial['test_rsme'], best_trial['T']]+[p['bs'],p['m_P'],p['m_factor'],p['sigma'],p['parametrize_Z'],p['use_all_m'],p['transformation']._get_name(),p['m_q_choice']])

            # data_list.append(
            #     [ds, f, best_trial['loss'], best_trial['test_loss'], best_trial['val_r2'], best_trial['test_r2']])
        except Exception as e:
            print(e)
    # all_hparam_best = pd.DataFrame(hparams_best_res,columns=['dataset','fold','bs','m_P','m_factor','sigma','parametrize_Z','use_all_m'])
    # all_hparam_best.to_csv(f'{results_folder}/all_hparam_best.csv')
    all_jobs = pd.DataFrame(data_list,columns=['dataset','fold','val_acc','NLL','val_r2','test_r2','rsme','T','bs','m_P','m_factor','sigma','parametrize_Z','use_all_m','act_func','m_q_choice'])
    all_jobs=all_jobs.sort_values(['dataset','fold'])
    all_jobs.to_csv(f'{results_folder}/all_jobs.csv')

    pm_all_jobs = pd.DataFrame(post_mortem,columns=['dataset','fold','hyp_idx','val_acc','NLL','val_r2','test_r2','rsme','T','bs','m_P','m_factor','sigma','parametrize_Z','use_all_m','act_func','m_q_choice'])
    pm_all_jobs=pm_all_jobs.sort_values(['dataset','fold','hyp_idx'])
    pm_all_jobs.to_csv(f'{results_folder}/pm_all_jobs.csv')

    summary_df_r2 = all_jobs.groupby(['dataset'])['test_r2'].mean()
    summary_df_std_r2 = all_jobs.groupby(['dataset'])['test_r2'].std()
    new_latex_df=summary_df_r2.apply(lambda x: rf'{round(x,2)} $\pm$')+summary_df_std_r2.apply(lambda x: f'{round(x,2)}')
    new_latex_df.to_latex(buf=f"{results_folder}/final_results_latex_r2.tex",escape=False)
    summary_df = all_jobs.groupby(['dataset'])['NLL'].mean()
    summary_df_std = all_jobs.groupby(['dataset'])['NLL'].std()
    new_latex_df=summary_df.apply(lambda x: rf'{round(x,2)}$\pm$')+summary_df_std.apply(lambda x: f'{round(x,2)}')
    new_latex_df = new_latex_df.sort_index( ascending=True,key=lambda col: col.str.lower())
    new_latex_df.to_latex(buf=f"{results_folder}/final_results_latex.tex",escape=False)


def extract_classification_data(results_folder):
    dataset = ['FashionMNIST','CIFAR10']
    folds=[0,1,2]
    p_list = list(itertools.product(dataset,folds))
    data_list =[]
    for (ds,f) in p_list:
        try:

            load_data_base_string_space = f'{results_folder}/{ds}_seed=0_fold_idx={f}_model=GWI/hparam_space.p'
            load_data_base_string = f'{results_folder}/{ds}_seed=0_fold_idx={f}_model=GWI/hyperopt_database.p'
            trials = pickle.load(open(load_data_base_string, "rb"))
            trials = dill.loads(trials)
            hps = pickle.load(open(load_data_base_string_space, "rb"))
            hparams = dill.loads(hps)

            jobs = []
            for idx,el in enumerate(trials.trials):
                if el['result']['status'] == 'ok':
                    jobs.append(el)
            best_trial = sorted(jobs, key=lambda x: x['result']['test_nll'], reverse=True)[-1]

            tmp = setup_hparams(best_trial)
            p = space_eval(hparams, tmp)
            best_trial=best_trial['result']
            print(best_trial['test_ood_auc'])
            print(best_trial['test_ood_auc_prior'])
            data_list.append([ds,f,best_trial['val_acc']*100.,best_trial['test_acc']*100.,best_trial['test_nll'],best_trial['test_ood_auc'],best_trial['test_ood_auc_prior'],best_trial['T']]+[p['bs'],p['m_P'],p['m_factor'],p['sigma'],p['parametrize_Z'],p['use_all_m'],p['width_x'],p['depth_fc'],p['transformation']])
        except Exception as e:
            print(e)
    all_jobs = pd.DataFrame(data_list,columns=['dataset','fold','val_acc','Accuracy','NLL','OOD-AUC','OOD-AUC PRIOR','T','bs','m_P','m_factor','sigma','parametrize_Z','use_all_m','width','depth_fc','transformation'])
    all_jobs.to_csv(f'{results_folder}/all_jobs.csv')



    summary_df = all_jobs.groupby(['dataset'])[['Accuracy','NLL','OOD-AUC','OOD-AUC PRIOR']].mean()#.reset_index()
    summary_df_std = all_jobs.groupby(['dataset'])[['Accuracy','NLL','OOD-AUC','OOD-AUC PRIOR']].std()#.reset_index()
    # print(summary_df)
    # print(summary_df.apply(lambda x: rf'${round(x,3)} \pm')
    new_latex_df=pd.DataFrame()

    for p in ['Accuracy','NLL','OOD-AUC','OOD-AUC PRIOR']:
        if p=='Accuracy':
            r=2
        else:
            r=3

        new_latex_df[p]=summary_df[p].apply(lambda x: rf'${round(x,r)} \pm')+summary_df_std[p].apply(lambda x: f'{round(x,2)}$')
    # print(summary_df[['Accuracy','NLL','OOD-AUC']])

    # print(new_latex_df)
    # print(summary_df_std)
    new_latex_df.to_latex(buf=f"{results_folder}/final_results_latex.tex",escape=False)


if __name__ == '__main__':
    pass
    # extract_regression_data('reg_mlp_True_results')
    # extract_regression_data('reg_mlp_True_Tanh_results')
    # extract_regression_data('reg_mlp_True_ReLU_results')

    # extract_regression_data('reg_mlp_False_Tanh_results')
    # extract_regression_data('reg_mlp_False_ReLU_results')
    # extract_regression_data('1000_reg_mlp_False_Tanh_results')
    # extract_regression_data('1000_reg_mlp_True_Tanh_results')
    # extract_regression_data('1000_reg_krr_False_Tanh_results')
    # extract_classification_data('100_1000_epoch_class_False_results')
    # extract_regression_data('50_1000_reg_krr_True_Tanh_results')
    # extract_regression_data('100_1000_reg_mlp_True_Tanh_results')
    # extract_regression_data('100_1000_reg_mlp_False_Tanh_results')
    extract_regression_data('new_100_1000_reg_mlp_False_Tanh_results')
    # extract_regression_data('100_1000_reg_krr_True_Tanh_results')
