import koopomics as ko
import pandas as pd
import torch

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Load Dataset
pregnancy_df = pd.read_csv('./input_data/pregnancy/pregnancy_interpolated_50M_robust_minmax_scaled_outlrem_uniform.csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

train_dataloader = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, 
                                      batch_size=10, max_Ksteps = 1)
test_dataloader = ko.OmicsDataloader(test_set_df, feature_list, replicate_id,
                                     batch_size=10, max_Ksteps = 1)

runconfig = ko.RunConfig()
runconfig.num_metabolites = 50
runconfig.feature_selected=True

# Load Model
embedding_model = ko.FF_AE([50,20,20,20,20,20,20,10], [10,20,20,20,20,20,20,50],E_dropout_rates= [0,0,0,0,0,0,0,0],activation_fn='leaky_relu')
operator_model = ko.InvKoop(latent_dim=10, reg=None,activation_fn='leaky_relu')

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)
baseline = ko.NaiveMeanPredictor(train_set_df, feature_list, mask_value=-2)

# Run training loop
TestingKoopnondelay.fit(train_dataloader, test_dataloader, runconfig=runconfig,
                         num_epochs = 100, lr=0.001, max_Kstep=1,
                         loss_weights = [1,0.5,0.00001,0.00001,0.01,0], mask_value=-2,
                         model_name = 'TestingKoopNonDelay_lrelu_M264', wandb_log=True,
                        learning_rate_change=0.2,
                        decayEpochs=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 410, 440, 470, 500],
                         baseline=baseline)
