import codecs
import numpy as np
import torch

#significant part of this function was got from https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/utils.py
def read_corpus(file):

    lines = list()
    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()

    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels



def data_word2idx(sentences):
    word_to_idx = dict()
    for sent in sentences:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    return word_to_idx


def data_ner_to_idx(ner_data):
    ner_to_idx = dict()
    for item in ner_data:
        for ner in item:
            if ner not in ner_to_idx:
                ner_to_idx[ner] = len(ner_to_idx)

    return ner_to_idx


def load_embedding_dictionary(emb_file):
    emb_dict = dict()
    for line in open(emb_file, 'r'):
        line = line.split(' ')
        word = line[0]
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        emb_dict[word] = vector

    return emb_dict


def embeding_to_tesnor_matrix(embedding_dict, word_to_idx_dict, convert_upper_to_lower_case=False):
    vocab_size = len(word_to_idx_dict)
    vector_size = len(next(iter(embedding_dict.values())))

    result = torch.zeros(vocab_size, vector_size, dtype=torch.float32)

    for word, idx in word_to_idx_dict.items():
        if word in embedding_dict:
            result[idx] = torch.tensor(embedding_dict[word])
        elif convert_upper_to_lower_case and word.lower() in embedding_dict:
            result[idx] = torch.tensor(embedding_dict[word.lower()])
        else:
            result[idx] = torch.tensor(embedding_dict["unk"])

    return result


def convert_to_indices_format(values, val_to_idx_dict):
    size = len(values)
    values_idx = torch.zeros(size, dtype=torch.long)
    for i in range(0, size):
        values_idx[i] = val_to_idx_dict[values[i]]

    return values_idx


def hello():
    print("hello")




#
#
# test_features, test_labels = read_corpus("test.txt")
#
# features_word_to_idx = data_word2idx(test_features)
# ner_to_idx = data_ner_to_idx(test_labels)
#
# a = 10










