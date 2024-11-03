import koopomics as ko
import pandas as pd
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Load Dataset
pregnancy_df = pd.read_csv('./input_data/pregnancy/pregnancy_interpolated_264M_robust_minmax_scaled_outlrem_uniform.csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

train_dataloader = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, 
                                      batch_size=5, max_Ksteps = 5)
test_dataloader = ko.OmicsDataloader(test_set_df, feature_list, replicate_id,
                                     batch_size=5, max_Ksteps = 5)

runconfig = ko.RunConfig()

# Load Model
embedding_model = ko.FF_AE([264,2000,2000,100], [100,2000,2000,264],E_dropout_rates= [0,0,0,0],activation_fn='leaky_relu')
operator_model = ko.InvKoop(latent_dim=100, reg='nondelay',activation_fn='leaky_relu')

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)
baseline = ko.NaiveMeanPredictor(train_set_df, feature_list, mask_value=-2)

# Run training loop
TestingKoopnondelay.fit(train_dataloader, test_dataloader, runconfig=runconfig,
                         num_epochs = 600, lr=0.001, max_Kstep=5,
                         loss_weights = [1,0.5,1,1,0.01,1], mask_value=-2,
                         model_name = 'TestingKoopNonDelay_lrelu_M264', wandb_log=True,
                        learning_rate_change=0.2,
                        decayEpochs=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 410, 440, 470, 500],
                         baseline=baseline)
