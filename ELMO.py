import torch

from elmo.models import ELMo
from elmo.dataset import NewsClassificationDataset

trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embeddingSize = int(input('Enter embedding size: '))
batchSize = int(input('Enter batch size: '))
learningRate = float(input('Enter learning rate: '))
epochs = int(input('Enter epochs: '))
retrain = input("Do you want to retrain the model? (y/n): ")
if retrain.lower() == 'y':
	retrain = True
else:
	retrain = False

resumeFromCheckpoint = False
if not retrain:
	resumeFromCheckpoint = input("Do you want to resume training from checkpoint? (y/n): ")
	if resumeFromCheckpoint.lower() == 'y':
		resumeFromCheckpoint = True

elmoModel = ELMo(embeddingSize=embeddingSize,
				 trainDataset=trainDataset,
				 device=device)

elmoModel.preTrain(batchSize=batchSize,
				   learningRate=learningRate,
				   epochs=epochs,
				   retrain=retrain,
				   resumeFromCheckpoint=resumeFromCheckpoint)
