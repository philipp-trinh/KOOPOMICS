import koopomics as ko
import pandas as pd

# Load Dataset
pregnancy_df = pd.read_csv('/Users/daviddornig/Documents/Master_Thesis/Bioinf/Code/philipp-trinh/KOOPOMICS/input_data/pregnancy/pregnancy_interpolated_264M_robust_minmax_scaled_outlrem_uniform.csv')

condition_id = 'Condition'
time_id = 'Gestational age (GA)/weeks'
replicate_id = 'Subject ID'
feature_list = pregnancy_df.columns[7:]
num_features = len(feature_list)

train_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Discovery'].copy()
test_set_df = pregnancy_df[pregnancy_df['Cohort'] == 'Validation (Test Set 1)'].copy()

train_dataloader = ko.OmicsDataloader(train_set_df, feature_list, replicate_id, time_id, 
                                      batch_size=5, max_Ksteps = 10)
test_dataloader = ko.OmicsDataloader(test_set_df, feature_list, replicate_id, time_id, 
                                     batch_size=5, max_Ksteps = 10)

# Load Model
embedding_model = ko.FF_AE([264,1000,1000,10], [10,1000,1000,264],E_dropout_rates= [0,0,0,0],activation_fn='tanh')
operator_model = ko.InvKoop(latent_dim=10, reg='nondelay')

TestingKoopnondelay = ko.KoopmanModel(embedding=embedding_model, operator=operator_model)


# Run training loop
ko.train(TestingKoopnondelay, train_dataloader, test_dataloader,
         lr= 0.001, learning_rate_change=0.8, loss_weights=[1,1,1,1,0.001,0],
         num_epochs=300, decayEpochs=[3, 6, 9, 12, 15, 18, 21, 24, 27],
         weight_decay=0, gradclip=1, max_Kstep=10, mask_value=-2,
         print_batch_info=False, model_name='TestingKoop264M_tanh_nondelay')
