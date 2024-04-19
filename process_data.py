from elmo.dataset import NewsClassificationDataset

trainDataset = NewsClassificationDataset('./data/News Classification Dataset/train.csv')
testDataset = NewsClassificationDataset('./data/News Classification Dataset/test.csv')