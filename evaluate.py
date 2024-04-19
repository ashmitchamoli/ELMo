from elmo.models import ELMoClassifier
from elmo.dataset import NewsClassificationDataset

import torch

trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')
testDataset = NewsClassificationDataset('../data/News Classification Dataset/test.csv', trueVocab=trainDataset.vocabulary)