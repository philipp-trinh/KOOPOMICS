import koopomics as ko
import pandas as pd
import torch
import wandb

# Load Dataset
#pregnancy_df = pd.read_csv('../input_data/pregnancy/pregnancy_interpolated_50M_median_centered_uniform_mask(-1e-9).csv')

pea_df = pd.read_csv('/lisc/user/trinh/KOOPOMICS/philipp-trinh/KOOPOMICS/input_data/pea_fungal/pea_247M_interpolated_normalized.csv')

condition_id = 'Treatment'
time_id = 'Dpi'
replicate_id = 'Plant_ID'

feature_list = pea_df.columns[6:]

pea_df['time_id'] = pea_df.groupby(time_id).ngroup()

time_id = 'time_id'
mask_value = -1e-9

#train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
#test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

#embedding_param_path = '../LISC_training/bestparams/Koop_embedding_parameters_run_xhb2hi1k.pth'
#em_param_path=embedding_param_path
hypmanager = ko.HypManager(pea_df, condition_id, replicate_id, time_id, feature_list, mask_value=-1e-9, fit=True) 

wandb.agent("elementar1-university-of-vienna/PregnancyKoop/9ndzr7c7", function=hypmanager.hyptrain, count=10)
