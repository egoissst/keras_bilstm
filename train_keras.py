import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import data_helpers
from gensim.models import Word2Vec
import pickle
from numpy import zeros
import os, time
import configparser
import logging

###############################################################################################

CONFIG_FILE_LOCATION = "./config_model.ini"
LOGGING_LEVEL = logging.DEBUG

###############################################################################################

clean_str_function = data_helpers.clean_str_pure_hindi
#clean_str_function = data_helpers.clean_str_updated

config = configparser.ConfigParser()
config.read(CONFIG_FILE_LOCATION)

word2vec_model_loc = config['FILE_LOCATION']['Word2Vec_Model']

positive_examples_loc = config['FILE_LOCATION']['PositiveDataFile']
negative_examples_loc = config['FILE_LOCATION']['NegativeDataFile']

embed_size = int(config['HYPERPARAMETERS']['EmbedSize'])
num_epochs = int(config['HYPERPARAMETERS']['NumEpochs'])
batch_size = int(config['HYPERPARAMETERS']['BatchSize'])

embed_trainable = config['HYPERPARAMETERS']['EmbedTrainable']

readme_info = config['EXTRAS']['ReadMeInfo']

###############################################################################################

POSITIVE_LABEL_VAL = 1
NEGATIVE_LABEL_VAL = 0

timestamp = str(int(time.time()))
base_folder_loc = 'saved_models/lstm_{}/'.format(timestamp)

readme_file_location = '{}readme.txt'.format(base_folder_loc)

model_weight_location = '{}model_weights.h5'.format(base_folder_loc)
model_arch_location = '{}model_architecture.json'.format(base_folder_loc)

tokenizer_pickle_loc = '{}tokenizer.pickle'.format(base_folder_loc)
pad_max_len_pickle_loc = '{}sent_padding_max_len.pickle'.format(base_folder_loc)

#create the model folder if it doesn't exist
if not os.path.exists(base_folder_loc):
    #logging.info('creating the model folder : {}'.format(base_folder_loc))
    os.makedirs(base_folder_loc)

#log settings
logging_file_loc = '{}model_training.log'.format(base_folder_loc)
str_format, datefmt='%(asctime)s: %(levelname)s: %(message)s', '%m/%d/%Y %I:%M:%S %p'
root_logger= logging.getLogger()
root_logger.setLevel(LOGGING_LEVEL)
handler = logging.FileHandler(logging_file_loc, 'w', 'utf-8')
handler.setFormatter = logging.Formatter(str_format, datefmt)
root_logger.addHandler(handler)

#logging each entry of the model_config file
for each_section in config.sections():
    logging.info('{}'.format(each_section))
    for (each_key, each_val) in config.items(each_section):
        logging.info('{} : {}'.format(each_key, each_val))

#writing readme info in the readme file
with open(readme_file_location, "w") as readme_text_file:
    logging.info('creating and writing to readme file : {}'.format(readme_file_location))
    logging.info('readme info : {}'.format(readme_info))
    readme_text_file.write(readme_info)

# Load data from files
positive_examples = list(open(positive_examples_loc, "r", encoding='utf-8').readlines())
positive_examples = [s.strip() for s in positive_examples]
positive_examples_label = len(positive_examples)*[POSITIVE_LABEL_VAL]

negative_examples = list(open(negative_examples_loc, "r", encoding='utf-8').readlines())
negative_examples = [s.strip() for s in negative_examples]
negative_examples_label = len(negative_examples)*[NEGATIVE_LABEL_VAL]

df = pd.DataFrame({'v3': positive_examples + negative_examples, 'v1':positive_examples_label + negative_examples_label})
df['v2'] = df['v3'].apply(lambda x: clean_str_function(x))

X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

#max number of words in a sentence
max_len = max([len(x.split(' ')) for x in X])

tok = Tokenizer(num_words=None)

pre_trained_word2vec_model = Word2Vec.load(word2vec_model_loc)
list_vocab = list(pre_trained_word2vec_model.wv.vocab)
combined_list = list_vocab + X_train.tolist()
#combined_list = X_train.tolist()
tok.fit_on_texts(combined_list)

# pickling tokenizer
with open(tokenizer_pickle_loc, 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

#pickling max_len of the input sentence
with open(pad_max_len_pickle_loc, 'wb') as handle:
    pickle.dump(max_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

max_words = len(tok.word_index) + 1

embedding_matrix = zeros((max_words, embed_size))
for word, i in tok.word_index.items():
    try:
        embedding_vector = pre_trained_word2vec_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        logging.info('word "{}" not found in word2vec'.format(word))
        pass

logging.info('embedding matrix of size {}*{} created'.format(max_words, embed_size))

sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


def get_model():
    inp = Input(shape=(max_len, ))
    x = Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=embed_trainable)(inp)    
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

#RNN model
#model = RNN()
#model.summary()
#model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
#          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
#          validation_split=0.2, verbose=2)

model = get_model()

#class_weight = {POSITIVE_LABEL_VAL: 1.,
#                NEGATIVE_LABEL_VAL: 20.}

model.fit(sequences_matrix,Y_train,batch_size=batch_size, epochs=num_epochs,
          validation_data=(test_sequences_matrix, Y_test), verbose=2)

#model.fit(sequences_matrix,Y_train,batch_size=batch_size, epochs=num_epochs,
#          validation_data=(test_sequences_matrix, Y_test), verbose=2, class_weight=class_weight)

#test_sequences = tok.texts_to_sequences(X_test)
#test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#accr = model.evaluate(test_sequences_matrix,Y_test)
#logging.info('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# Save the weights
#model_weight_location = 'saved_models/model_weights_bilstm_w2v_hin.h5'
model.save_weights(model_weight_location)
logging.info('saving the model weights at {}'.format(model_weight_location))

# Save the model architecture
#model_arch_location = 'saved_models/model_architecture_bilstm_w2v_hin.json'
with open(model_arch_location, 'w') as f:
    f.write(model.to_json())
    logging.info('saving the model architecture at {}'.format(model_arch_location))

