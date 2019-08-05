import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import utils

# the idea of this class was got from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
class LSTMTagger(nn.Module):
    def __init__(self, word_embeddings, hidden_dim, target_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = word_embeddings
        self.embedding_dim = word_embeddings.weight.shape[1]

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class Trainer:
    def __init__(self, model, loss_function, optimizer, feature_to_idx_dict, target_to_idx_dict):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.feature_to_idx_dict = feature_to_idx_dict
        self.target_to_idx_dict = target_to_idx_dict

    def train(self, features, targets, num_epoch):
        assert len(features) == len(targets)

        size = len(features)

        for epoch in range(num_epoch):
            for i in range(0, size):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                feature_idx = utils.convert_to_indices_format(features[i], self.feature_to_idx_dict)
                target_idx = utils.convert_to_indices_format(targets[i], self.target_to_idx_dict)

                # Step 3. Run our forward pass.
                target_scores = self.model(feature_idx)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = self.loss_function(target_scores, target_idx)
                loss.backward()
                self.optimizer.step()


    def predict(self, features):
        all_pred_scores = []
        with torch.no_grad():

            for i in range(0, len(features)):

                inputs = utils.convert_to_indices_format(features[i], self.feature_to_idx_dict)
                scores = self.model(inputs)

                all_pred_scores.append(scores)

        return all_pred_scores





