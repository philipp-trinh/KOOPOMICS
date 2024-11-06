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

batch_size = 42
max_Kstep=1
dl_structure = 'random'
mask_value=-1e-9

train_dataloader = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, 
                                      batch_size=batch_size, dl_structure=dl_structure,
                                      max_Kstep = max_Kstep, mask_value=mask_value)
test_dataloader = ko.OmicsDataloader(test_set_df, feature_list, replicate_id,
                                     batch_size=batch_size, dl_structure=dl_structure,
                                     max_Kstep = max_Kstep, mask_value=mask_value)

runconfig = ko.RunConfig()
runconfig.num_metabolites = 264
runconfig.feature_selected=False
runconfig.batch_size = batch_size
runconfig.dl_structure = dl_structure


# Load Model
embedding_model = ko.FF_AE([264,2000,100], [100,2000,264],E_dropout_rates= [0,0,0],activation_fn='leaky_relu')
operator_model = ko.LinearizingKoop(linearizer=ko.FFLinearizer([100,2000,100], [100,2000,100], activation_fn='leaky_relu'), koop=ko.InvKoop(latent_dim=100, reg='skewsym', activation_fn='leaky_relu'))

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)
baseline = ko.NaiveMeanPredictor(train_set_df, feature_list, mask_value=mask_value)

embedding_param_path = './bestparams/Koop_embedding_parameters_run_xhb2hi1k.pth'
# Run training loop
TestingKoopnondelay.modular_fit(train_dataloader, test_dataloader, runconfig=runconfig,
                         num_epochs = 164, learning_rate=0.00011, start_Kstep=0, max_Kstep=max_Kstep,
                         loss_weights = [1,0.5,1,1,0.01,0], mask_value=mask_value,
                         model_name = 'TestingKoopNonDelay_lrelu_M264', use_wandb=True,
                        learning_rate_change=0.99, early_stop=False, patience=20,
                         decayEpochs=[40,100,200], embedding_param_path=embedding_param_path,
                         baseline=baseline, grad_clip=0, weight_decay=0)
