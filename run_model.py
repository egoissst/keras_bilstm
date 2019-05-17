import pickle
from keras.models import model_from_json
from keras.optimizers import RMSprop
import pandas as pd
import data_helpers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

###############################################################################################

model_folder = "lstm_1558011454"

#clean_str_function = data_helpers.clean_str_updated
clean_str_function = data_helpers.clean_str_pure_hindi

#test_file_location = 'testing_data/TOI_2_May.csv'
testing_file_name = 'NBTO_14_15_May_1551860056_1558012024_prediction'
testing_file_suffix = 'csv'

spam_threshold = 0.5

###############################################################################################

test_file_location = 'testing_data/{}.{}'.format(testing_file_name, testing_file_suffix)
output_file_location = 'output_run/{}_{}.{}'.format(testing_file_name, model_folder, testing_file_suffix)

base_folder_loc = 'saved_models/{}/'.format(model_folder)

model_weight_location = '{}model_weights.h5'.format(base_folder_loc)
model_arch_location = '{}model_architecture.json'.format(base_folder_loc)

tokenizer_pickle_loc = '{}tokenizer.pickle'.format(base_folder_loc)
pad_max_len_pickle_loc = '{}sent_padding_max_len.pickle'.format(base_folder_loc)

print('started model running for test file : {}'.format(test_file_location))
print('using model : {}'.format(base_folder_loc))

# load json and create model
json_file = open(model_arch_location, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_weight_location)
print("Loaded model from disk")

# evaluate loaded model on test data

df = pd.read_csv(test_file_location)

eval_set = df['C_T'].tolist()
eval_set = [clean_str_function(x) for x in eval_set]

with open(tokenizer_pickle_loc, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(pad_max_len_pickle_loc, 'rb') as handle:
    max_len = pickle.load(handle)

sequences = tokenizer.texts_to_sequences(eval_set)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

keras_pred = loaded_model.predict(sequences_matrix)
df['bilstm_prob'] = keras_pred
df['bilstm_pred'] = df['bilstm_prob'].apply(lambda x: x>spam_threshold)

df.to_csv(output_file_location, index=False)
print('file writing completed at loc : {}'.format(output_file_location))

#df
