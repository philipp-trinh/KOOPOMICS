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

batch_size = 10
max_Ksteps=1
dl_structure = 'random'

train_dataloader = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, 
                                      batch_size=batch_size, dl_structure=dl_structure, max_Ksteps = max_Ksteps)
test_dataloader = ko.OmicsDataloader(test_set_df, feature_list, replicate_id,
                                     batch_size=batch_size, dl_structure=dl_structure, max_Ksteps = max_Ksteps)

runconfig = ko.RunConfig()
runconfig.num_metabolites = 50
runconfig.feature_selected=True
runconfig.batch_size = batch_size
runconfig.dl_structure = dl_structure


# Load Model
embedding_model = ko.FF_AE([50,2000,20], [20,2000,50],E_dropout_rates= [0,0.3,0],activation_fn='leaky_relu')
operator_model = ko.InvKoop(latent_dim=20, reg='banded', bandwidth=5, activation_fn='leaky_relu')

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)
baseline = ko.NaiveMeanPredictor(train_set_df, feature_list, mask_value=-2)

# Run training loop
TestingKoopnondelay.fit_embedding(train_dataloader, test_dataloader, runconfig=runconfig,
                         num_epochs = 200, lr=0.000001, max_Kstep=max_Ksteps,
                         loss_weights = [1,0.5,1,1,0.01,0], mask_value=-2,
                         model_name = 'TestingKoopNonDelay_lrelu_M50', use_wandb=True,
                        learning_rate_change=0.8, early_stop=True,
                        decayEpochs=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 410, 440, 470, 500],
                         baseline=baseline)
