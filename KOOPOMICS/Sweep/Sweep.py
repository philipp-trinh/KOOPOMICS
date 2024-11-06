import koopomics as ko
import pandas as pd
import torch
import wandb

# Load Dataset
pregnancy_df = pd.read_csv('../input_data/pregnancy/pregnancy_interpolated_264M_median_centered_uniform_mask(-1e-9).csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)
mask_value = -1e-9

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

hypmanager = ko.HypManager(train_set_df, test_set_df, condition_id, replicate_id, time_id, feature_list, mask_value=-1e-9, embedding_fit=True) 

wandb.agent("elementar1-university-of-vienna/PregnancyKoop/niy9htfz", function=hypmanager.hyptrain, count=10)
