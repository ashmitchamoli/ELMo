import torch

from elmo.models import ELMoClassifier
from elmo.dataset import NewsClassificationDataset

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(type(device), device)

	trainDataset = NewsClassificationDataset('./data/News Classification Dataset/train.csv')
	testDataset = NewsClassificationDataset('./data/News Classification Dataset/test.csv', trueVocab=trainDataset.vocabulary)

	elmoClassifier = ELMoClassifier(512, 						# embedding size
								 	trainDataset,				# train dataset
									device,						# device
									combineMode='wsum',
									hiddenEmbeddingSize=256,
									numLayers=2,
									hiddenSizes=[128, 64],
									activation='relu')
	 
	elmoClassifier.preTrainElmo(testDataset,
								batchSize=16, 
						 		learningRate=0.005, 
								epochs=16,
								# resumeFromCheckpoint=True,
								retrain=True)

	elmoClassifier.train(batchSize=32,
					  	 epochs=10,
						 retrain=True,
						 devDataset=testDataset,
						#  resumeFromCheckpoint=True,
						 learningRate=0.0005)
	
	metrics = elmoClassifier.evaluate(trainDataset)
	print("Train Report:")
	print(metrics['report'])

	testMetrics = elmoClassifier.evaluate(testDataset)
	
	print("\nTest Report:")
	print(testMetrics['report'])