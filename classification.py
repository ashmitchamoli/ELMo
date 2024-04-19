import torch

from elmo.models import ELMoClassifier
from elmo.dataset import NewsClassificationDataset

trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embeddingSize = int(input('Enter embedding size: '))
combineMode = input('Enter combine mode (function, wsum, sum): ')
hiddenEmbeddingSize = int(input('Enter hidden embedding size: '))
numLayers = int(input('Enter number of layers: '))
hiddenSizes = list(map(int, input('Enter hidden sizes (comma-separated): ').split(',')))
activation = input('Enter activation (tanh, relu): ')
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

elmoClassifier = ELMoClassifier(embeddingSize, 				# embedding size
								trainDataset,				# train dataset
								device,						# device
								combineMode=combineMode,
								hiddenEmbeddingSize=hiddenEmbeddingSize,
								numLayers=numLayers,
								hiddenSizes=hiddenSizes,
								activation=activation)

elmoClassifier.train(batchSize=batchSize,
					 epochs=epochs,
					 learningRate=learningRate,
					 retrain=retrain,
					 resumeFromCheckpoint=resumeFromCheckpoint)
