import koopomics as ko
import pandas as pd
import torch

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Load Dataset
pregnancy_df = pd.read_csv('../input_data/pregnancy/pregnancy_interpolated_264M_median_centered_uniform_mask(-1e-9).csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

# Step 1: Sort unique timepoints and define training and testing timepoints
unique_timepoints = sorted(pregnancy_df[time_id].unique())
# Select the first half for training and the second half for testing
split_index = int(len(unique_timepoints) * 0.8)
train_timepoints = unique_timepoints[:split_index]
test_timepoints = unique_timepoints[split_index:]

# Step 2: Split train and test dataframes based on time_id values
#train_set_df = pregnancy_df[pregnancy_df[time_id].isin(train_timepoints)]
#test_set_df = pregnancy_df[pregnancy_df[time_id].isin(test_timepoints)]


batch_size = 700
max_Kstep=5
dl_structure = 'temporal'
mask_value=-1e-9

train_dl = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, 
                                      batch_size=batch_size, dl_structure=dl_structure,
                                      max_Kstep = max_Kstep, mask_value=mask_value, delay_size=3)
train_dataloader = train_dl.get_dataloaders()

test_dl = ko.OmicsDataloader(test_set_df, feature_list, replicate_id,
                                     batch_size=600, dl_structure=dl_structure,
                                     max_Kstep = max_Kstep, mask_value=mask_value, delay_size=3)
test_dataloader = test_dl.get_dataloaders()


#train_dataloader, test_dataloader = ko.OmicsDataloader(pregnancy_df, feature_list, replicate_id,
                                     #batch_size=batch_size, dl_structure=dl_structure,
                                     #max_Kstep = max_Kstep, mask_value=mask_value, train_ratio=0.7)

runconfig = ko.RunConfig()
runconfig.num_metabolites = 264
runconfig.feature_selected=False
runconfig.batch_size = batch_size
runconfig.dl_structure = dl_structure


# Load Model
embedding_model = ko.FF_AE([264,2000,150], [150,2000,264],E_dropout_rates= [0,0,0],activation_fn='leaky_relu')
operator_model = ko.LinearizingKoop(linearizer=ko.FFLinearizer([150,2000,150], [150,2000,150], activation_fn='leaky_relu'), koop=ko.InvKoop(latent_dim=150, reg='nondelay', activation_fn='leaky_relu'))
#operator_model = ko.InvKoop(latent_dim=150, reg='nondelay', activation_fn='leaky_relu')

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)
baseline = ko.NaiveMeanPredictor(train_set_df, feature_list, mask_value=mask_value)

embedding_param_path = './bestparams/Koop_embedding_parameters_run_xhb2hi1k.pth'
# Run training loop
TestingKoopnondelay.modular_fit(train_dataloader, test_dataloader, runconfig=runconfig,
                         num_epochs = 3000, learning_rate=0.0001, start_Kstep=0, max_Kstep=max_Kstep,
                         loss_weights = [1,0,1,1,0,0], mask_value=mask_value,
                         model_name = 'TestingKoopNonDelay_lrelu_M264', use_wandb=False,
                        batch_verbose=False,
                        learning_rate_change=0.8, early_stop=True, patience=20,
                         baseline=baseline, grad_clip=0, weight_decay=1e-4)
