import torch
import utils
import pandas as pd
import numpy as np


class Metric:

    def __init__(self, ner_to_idx_dict) :
        self.ner_to_idx_dict = ner_to_idx_dict
        self.micro_avg_precision = 0
        self.micro_avg_recall = 0
        self.micro_avg_f1 = 0
        self.micro_avg_f_half = 0


    def calc_confusion_matrix(self, pred_sentences_scores, targets):
        assert len(pred_sentences_scores) == len(targets)
        size = len(pred_sentences_scores)

        binary_pred_sentences_scores = []
        binary_targets = []


        for i in range(0, size):
            binary_pred_sentences_scores.append(Metric._sentence_scores_to_binary(pred_sentences_scores[i]))
            binary_targets.append(utils.convert_to_indices_format(targets[i], self.ner_to_idx_dict))

        ner_tp_fp_fn = self._calc_tp_fp_fn(binary_pred_sentences_scores, binary_targets)


        self._calc_micro_avg(ner_tp_fp_fn)
        precision, recall, f1, f_half = self._calc_precision_recall_f1_f_half(ner_tp_fp_fn)

        df = self._create_data_frame(precision, recall, f1, f_half)

        return df


    def _calc_micro_avg(self, ner_tp_fp_fn):
        tp = [item[0] for item in ner_tp_fp_fn]
        fp = [item[1] for item in ner_tp_fp_fn]
        fn = [item[2] for item in ner_tp_fp_fn]


        self.micro_avg_precision = sum(tp) / (sum(tp) + sum(fp))
        self.micro_avg_recall = sum(tp) / (sum(tp) + sum(fn))

        self.micro_avg_f1 = self._calc_f_beta_score(self.micro_avg_precision, self.micro_avg_recall)
        self.micro_avg_f_half = self._calc_f_beta_score(self.micro_avg_precision, self.micro_avg_recall, beta=0.5)


    @staticmethod
    def _sentence_scores_to_binary(sentence_scores):
        binary_pred = torch.zeros(len(sentence_scores), dtype=torch.long)
        for i in range(0, len(sentence_scores)):
            max_idx = sentence_scores[i].argmax()
            binary_pred[i] = max_idx

        return binary_pred


    def _calc_tp_fp_fn(self, binary_pred_sentences_scores, binary_targets):
        size = len(binary_pred_sentences_scores)

        ner_tp_fp_fn = np.zeros((len(self.ner_to_idx_dict), 3))

        for i in range(0, size):
            for j in range(0, len(binary_pred_sentences_scores[i])):
                true_ner_idx = binary_targets[i][j].item()
                pred_ner_idx = binary_pred_sentences_scores[i][j].item()
                if pred_ner_idx == true_ner_idx:
                    #add true positive
                    ner_tp_fp_fn[true_ner_idx][0] += 1
                else:
                    #add false positive
                    ner_tp_fp_fn[pred_ner_idx][1] += 1

                    #add false negative
                    ner_tp_fp_fn[true_ner_idx][2] += 1

        return ner_tp_fp_fn


    def _calc_precision_recall_f1_f_half(self, ner_tp_fp_fn):
        precision = [item[0] / (item[0] + item[1]) for item in ner_tp_fp_fn]
        recall = [item[0] / (item[0] + item[2]) for item in ner_tp_fp_fn]

        size = len(ner_tp_fp_fn)
        f1 = size * [0]
        f_half = size * [0]

        for i in range(0, size):
            f1[i] = self._calc_f_beta_score(precision[i], recall[i])
            f_half[i] = self._calc_f_beta_score(precision[i], recall[i], beta=0.5)

        return precision, recall, f1, f_half

    def _calc_f_beta_score(self, precision, recall, beta=1.0):
        f = (1 + beta ** 2) * precision * recall / ((beta ** 2)*precision + recall)
        return f


    def _create_data_frame(self, precision, recall, f1, f_half):
        d = dict()
        d['NER'] = list(self.ner_to_idx_dict.keys()) + ['micro average']
        d["Precision"] = precision + [self.micro_avg_precision]
        d['Recall'] = recall + [self.micro_avg_recall]
        d["F1"] = f1 + [self.micro_avg_f1]
        d["F0.5"] = f_half + [self.micro_avg_f_half]

        return pd.DataFrame(d)











