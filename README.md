# ELMo
An implementation of ELMo architecture from scratch using PyTorch. Refer to the report for detailed analysis. 

## Introduction
ELMo is a deep contextualized word representation that models both complex characteristics of word use (e.g., syntax and semantics), and how these uses vary across linguistic contexts (Peters et al., 2018).

## Usage


## Hyperparameter Tuning
Training for appropriate number of epochs.
- batchSize 32
- learningRate 0.001
- hiddenEmebddingSize = 256
- hiddenSizes = [128, 64]
- numLayers = 2
Metrics for sum:
	Accuracy: 0.9089473684210526
	F1: 0.9088496639644189
	Precision: 0.9089044779936902
	Recall: 0.9089473684210525
	Report:               precision    recall  f1-score   support

           0       0.93      0.90      0.92      1900
           1       0.95      0.97      0.96      1900
           2       0.87      0.88      0.88      1900
           3       0.89      0.88      0.89      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600

Metrics for wsum:
	Accuracy: 0.9090789473684211
	F1: 0.9090398697163773
	Precision: 0.9090290183385228
	Recall: 0.9090789473684211
	Report:               precision    recall  f1-score   support

           0       0.91      0.92      0.92      1900
           1       0.97      0.97      0.97      1900
           2       0.87      0.86      0.87      1900
           3       0.88      0.88      0.88      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600

Metrics for function:
	Accuracy: 0.9082894736842105
	F1: 0.9085069325412154
	Precision: 0.9091382495039095
	Recall: 0.9082894736842105
	Report:               precision    recall  f1-score   support

           0       0.92      0.91      0.91      1900
           1       0.97      0.95      0.96      1900
           2       0.88      0.86      0.87      1900
           3       0.86      0.91      0.88      1900

    accuracy                           0.91      7600
   macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600
