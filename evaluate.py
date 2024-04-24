from elmo.models import ELMoClassifier
from elmo.dataset import NewsClassificationDataset

import torch
import matplotlib.pyplot as plt
import seaborn as sns

trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')
testDataset = NewsClassificationDataset('../data/News Classification Dataset/test.csv', trueVocab=trainDataset.vocabulary)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

combineModes = ['function', 'wsum', 'sum']

for combineMode in combineModes:
	model = ELMoClassifier(512,
						   trainDataset,
						   device,
						   combineMode=combineMode,
						   hiddenEmbeddingSize=256,
						   numLayers=2,
						   hiddenSizes=[128, 64],
						   activation='relu')
	
	model.preTrainElmo(batchSize=32, learningRate=0.005, epochs=10, retrain=False)
	model.train(batchSize=32, epochs=10, learningRate=0.005, retrain=False, devDataset=testDataset)

	metrics = model.evaluate(testDataset, verbose=False)

	print(f"""
Metrics for {combineMode}:
	Accuracy: {metrics['accuracy']}
	F1: {metrics['f1']}
	Precision: {metrics['precision']}
	Recall: {metrics['recall']}
	Report: {metrics['report']}
""")
	
	# plot confusion matrix
	fig, ax = plt.subplots(figsize=(10, 10))
	sns.heatmap(metrics['confusion matrix'], annot=True, ax=ax)
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')

	fig.savefig(f'confusion_matrix_{combineMode}.png')
