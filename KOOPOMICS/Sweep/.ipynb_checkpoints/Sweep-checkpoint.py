import koopomics as ko
import pandas as pd
import torch

# Load Dataset
pregnancy_df = pd.read_csv('../input_data/pregnancy/pregnancy_interpolated_264M_robust_minmax_scaled_outlrem_uniform.csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

hypmanager = ko.HypManager(train_df, test_df, condition_id, replicate_id, time_id, feature_list) 

wandb.agent("elementar1-university-of-vienna/PregnancyKoop/mwr74va2", function=hypmanager.hyptrain, count=30)
