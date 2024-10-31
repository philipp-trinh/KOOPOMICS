import koopomics as ko
import pandas as pd
import torch

device = ko.get_device()
print(device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Load Dataset
pregnancy_df = pd.read_csv('./input_data/pregnancy/pregnancy_interpolated_50M_robust_minmax_scaled_outlrem_uniform.csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

train_dataloader = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, time_id, 
                                      batch_size=5, max_Ksteps = 5)
test_dataloader = ko.OmicsDataloader(test_set_df, feature_list, replicate_id, time_id, 
                                     batch_size=5, max_Ksteps = 5)

# Load Model
embedding_model = ko.FF_AE([50,2000,2000,20], [20,2000,2000,50],E_dropout_rates= [0,0,0,0],activation_fn='leaky_relu')
operator_model = ko.InvKoop(latent_dim=20, reg='nondelay')

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)
TestingKoopnondelay = TestingKoopnondelay.to(device)

# Run training loop
ko.train(TestingKoopnondelay, train_dataloader, test_dataloader,
         lr= 0.00001, learning_rate_change=0.2, loss_weights=[1,0.5,1,1,0.1,0],
         num_epochs=600, decayEpochs=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 410, 440, 470, 500],
         weight_decay=0, gradclip=0.05, max_Kstep=1, mask_value=-2,
         print_batch_info=False, model_name='TestingKoop264M_tanh_nondelay')
