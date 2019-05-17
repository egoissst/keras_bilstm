import numpy as np
import re
import itertools
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str_updated(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #replace html tags and character entities with space
    string = re.sub(r"(<br/>|&quot;|&amp;|&gt;|&lt;|&nbsp;|&ndash;|&ensp;|&mdash;|&lsquo;|&rsquo;|&rdquo;|&ldquo;|&bdquo;)", " ", string)
    string = re.sub(r"[^A-Za-z0-9!\'*$@#]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    #removing the single quotes
    string = re.sub(r"\'", "", string)

    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()

    #reducing multiple consecutive occurrences to two occurrences
    string = re.sub(r'(.)\1{2,}',r'\1\1', string)

    list_words = []
    for x in string.split(" "):
        #remove starting and ending exclamations (!)
        x = re.sub(r"^!+", "", x)
        x = re.sub(r"!+$", "", x)
        #remove starting hashes (#)
        x = re.sub(r"^#+", "", x)
        #remove those tokens that don't contain any alphabets
        if(string_contains_alphabet(x)):
            list_words.append(x)

    #rejoin the retrieved list to form a sentence
    string = " ".join(list_words)
    return string.strip().lower()


def string_contains_alphabet(string):
    if re.search('[a-zA-Z]', string):
        return True
    else:
        return False


def clean_str_old(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_pure_hindi(string):
    """ 
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^\u0900\u0901\u0902\u0903\u0904\u0905\u0906\u0907\u0908\u0909\u090a\u090b\u090c\u090d\u090e\u090f\u0910\u0911\u0912\u0913\u0914\u0915\u0916\u0917\u0918\u0919\u091a\u091b\u091c\u091d\u091e\u091f\u0920\u0921\u0922\u0923\u0924\u0925\u0926\u0927\u0928\u0929\u092a\u092b\u092c\u092d\u092e\u092f\u0930\u0931\u0932\u0933\u0934\u0935\u0936\u0937\u0938\u0939\u093a\u093b\u093c\u093d\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948\u0949\u094a\u094b\u094c\u094d\u094e\u094f\u0950\u0951\u0952\u0953\u0954\u0955\u0956\u0957\u0958\u0959\u095a\u095b\u095c\u095d\u095e\u095f\u0960\u0961\u0962\u0963\u0964\u0965\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f\u0970\u0971\u0972\u0973\u0974\u0975\u0976\u0977\u0978\u0979\u097a\u097b\u097c\u097d\u097e\u097f]", " ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def clean_str(string):
    """ 
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`*$@#]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
   # string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
   # string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
   # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    #reducing multiple consecutive occurrences to two occurrences
    #string = re.sub(r'([A-Za-z])\1{2,}',r'\1\1', string)
    string = re.sub(r'(.)\1{2,}',r'\1\1', string)    
    return string.strip().lower()

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    #positive_examples_loc = "./data/rt-polarity-hindi_uniq_shuf.pos"
    positive_examples_loc = "./data/rt-polarity-hindi_uniq_shuf.pos_oversampled_shuf"
    negative_examples_loc = "./data/rt-polarity-hindi_uniq_shuf.neg"    

    # Load data from files
    #positive_examples = list(open("./data/rt-polarity.pos").readlines())
    #positive_examples = list(open("./data/rt-polarity_myt.pos", "r", encoding='utf-8').readlines())
    positive_examples = list(open(positive_examples_loc, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    #negative_examples = list(open("./data/rt-polarity.neg").readlines())
    #negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    negative_examples = list(open(negative_examples_loc, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    #x_text = [clean_str(sent) for sent in x_text]
    x_text = [clean_str_pure_hindi(sent) for sent in x_text]   
 
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>", pad_length = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = pad_length if pad_length else max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab_old(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_vocab(sentences, word2vec_model=None):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    
    if word2vec_model!=None:
        list_vocab = list(word2vec_model.wv.vocab)
        vocabulary_inv = list(set(vocabulary_inv + list_vocab))

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary, labels=None, padding_word = "<PAD/>"):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    #x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary[padding_word] for word in sentence] for sentence in sentences])
    #y = np.array(labels) if labels!=None else None
    y = np.array(labels)
    return [x, y]


def load_data(word2vec_model=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded, word2vec_model)
    x, y = build_input_data(sentences_padded, vocabulary, labels = labels)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
