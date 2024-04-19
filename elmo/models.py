import torch
import os
from torch.utils.data import DataLoader
from typing import Literal
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from dataset import NewsClassificationDataset

MODEL_CHECKPOINTS_PATH = './model_checkpoints'

class ELMo(torch.nn.Module):
	"""
	ELMo model to obtain contextualized word embeddings.
	"""
	def __init__(self, embeddingSize : int, 
			  	 trainDataset : NewsClassificationDataset,
				 device : torch.device = torch.device('cpu')) -> None:
		"""
		Parameters:
			embeddingSize (int): The size of the embedding.
			vocabulary (bidict): A bidirectional dictionary containing the vocabulary and word indices.
			combineMode (Literal['function', 'wsum', 'sum']): The mode to combine embeddings.
				- 'function': A linear layer is used to combine embeddings.
				- 'wsum': Take the weighted sum of embeddings using trainable weights.
				- 'sum': Take the weighted sum of embeddings using fixed random weights.
		"""
		# ensure that embeddingSize is even
		assert embeddingSize % 2 == 0

		super(ELMo, self).__init__()

		self._modelSavePath = os.path.join(MODEL_CHECKPOINTS_PATH, 'ELMo')
		self._modelFileSuffix = f'_{embeddingSize}_{len(trainDataset.vocabulary)}'

		self.trainDataset = trainDataset
		self.vocabulary = trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)
		self.embeddingSize = embeddingSize
		self.device = device
		

		self.embeddings = torch.nn.Embedding(self.vocabSize, embeddingSize)
		self.biLM1 = torch.nn.LSTM(input_size=embeddingSize,
								   hidden_size=embeddingSize//2,
								   num_layers=1,
								   bidirectional=True,
								   batch_first=True)
		self.biLM2 = torch.nn.LSTM(input_size=embeddingSize,
							   	   hidden_size=embeddingSize//2,
								   num_layers=1,
								   bidirectional=True,
								   batch_first=True)
		self.preTrainingClassifier = torch.nn.Linear(embeddingSize, self.vocabSize)

	def nextWordPredictionForward(self, X : torch.Tensor) -> torch.Tensor:
		# get the embeddings
		e1 = self.embeddings(X)
		
		# get the output of the first LSTM
		e2, _ = self.biLM1(e1)

		# get the output of the second LSTM
		e3, _ = self.biLM2(e2)

		# residual connection
		e3 = (e3 + e2) / 2

		# prepare input
		input = torch.cat((torch.cat((torch.zeros((e3.shape[0], 1, self.embeddingSize//2)).to(self.device),
							  		  e3[:, :-1, :self.embeddingSize//2]), dim=1),
						   torch.cat((e3[:, 1:, self.embeddingSize//2:], 
				  					  torch.zeros((e3.shape[0], 1, self.embeddingSize//2)).to(self.device)), dim=1)), 
						  dim=2)
		input = input.view(-1, self.embeddingSize)
		
		# predict words
		wordPredictions = self.preTrainingClassifier(input)

		return wordPredictions

	def preTrain(self, batchSize : int = 16, 
			     learningRate : float = 0.001, 
				 epochs : int = 10, verbose : bool = True, 
				 retrain : bool = False,
				 resumeFromCheckpoint : bool = False) -> None:
		if not retrain and self.__loadModel__():
			if verbose:
				print('Model checkpoint loaded.')
			if not resumeFromCheckpoint:
				return
			else:
				if verbose:
					print('Resuming training from checkpoint.')
		else:
			if verbose:
				print('Model checkpoint not found or retrain flag is set. Training from scratch.')

		if verbose:
			print("Commencing pretraining...")
			
		trainLoader = DataLoader(self.trainDataset, batchSize, shuffle=True, collate_fn=self.trainDataset._custom_collate_)

		params = [ { 'params': self.biLM1.parameters() },
				   { 'params': self.biLM2.parameters() },
				   { 'params': self.embeddings.parameters() },
				   { 'params': self.preTrainingClassifier.parameters() } ]
		optimizer = torch.optim.Adam(params, lr=learningRate)
		
		criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocabulary['<PAD>'])

		epochMetrics = {}
		self.to(self.device)
		for epoch in range(epochs):
			runningLoss = 0
			epochMetrics[epoch] = {}
			for X, y in tqdm(trainLoader, desc='Training', leave=True):
				X = X.to(self.device)
				y = y.to(self.device)

				optimizer.zero_grad()

				predictions = self.nextWordPredictionForward(X)

				loss = criterion(predictions, X.view(-1))
				loss.backward()

				optimizer.step()

				runningLoss += loss.item()

			epochMetrics[epoch]['train loss'] = runningLoss / len(trainLoader)
			
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} loss {epochMetrics[epoch]['train loss']:.3f}")

			if (epoch+1) % 2 == 0:
				self.__saveModel__()
				if verbose:
					print(f'Epoch {epoch+1}/{epochs}: Model checkpoint saved.')
		
		self.__saveModel__()
		if verbose:
			print("Pretraining completed. Model weights saved\n")

	def __saveModel__(self) -> None:
		"""
		Save the current parameters of the model.
		"""
		if not os.path.exists(MODEL_CHECKPOINTS_PATH):
			os.mkdir(MODEL_CHECKPOINTS_PATH)

		if not os.path.exists(self._modelSavePath):
			os.mkdir(self._modelSavePath)
		
		torch.save(self.state_dict(), os.path.join(self._modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt'))

	def __loadModel__(self) -> bool:
		if not os.path.exists(os.path.join(self._modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt')):
			print(os.path.join(self._modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt'))
			return False
		
		self.load_state_dict(torch.load(os.path.join(self._modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt')))
		return True
	
	def evaluate(self, testDataset : NewsClassificationDataset) -> dict:
		testLoader = DataLoader(testDataset, 8, shuffle=False, collate_fn=testDataset._custom_collate_)

		epochMetrics = {}
		self.to(self.device)
		with torch.no_grad():
			for X, y in tqdm(testLoader, desc='Evaluating', leave=True):
				X = X.to(self.device)

				predictions = self.classificationForward(X)
				
				epochMetrics['accuracy'] = accuracy_score(y.view(-1).cpu(), predictions.argmax(dim=1).view(-1).cpu())

		return epochMetrics

class ELMoClassifier(torch.nn.Module):
	def __init__(self, *elmoParams, combineMode : Literal['function', 'wsum', 'sum'] = 'function', hiddenEmbeddingSize : int = 256, numLayers : int = 1, hiddenSizes : list[int] = [64], activation : Literal['tanh', 'relu', 'sigmoid'] = 'tanh') -> None:
		super(ELMoClassifier, self).__init__()

		self.ELMo = ELMo(*elmoParams)

		self._modelSavePath = os.path.join(MODEL_CHECKPOINTS_PATH, 'ELMo_classifier')
		self._classifierModelFileSuffix = self.ELMo._modelFileSuffix + f"_{hiddenEmbeddingSize}_{numLayers}_{hiddenSizes}_{activation}_{combineMode}"

		self.trainDataset = self.ELMo.trainDataset
		self.vocabulary = self.trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)
		self.classes = self.trainDataset.classes
		self.numClasses = self.trainDataset.numClasses
		self.embeddingSize = self.ELMo.embeddingSize
		self.device = self.ELMo.device

		self.numLayers = numLayers
		self.bidirectional = True
		self.hiddenEmbeddingSize = hiddenEmbeddingSize
		self.hiddenSizes = [ hiddenEmbeddingSize*(self.bidirectional+1) ] + hiddenSizes + [ self.numClasses ]
		self.combineMode = combineMode
		self.combineEmbeddings = None # takes a tuple of 3 embeddings and returns the final embedding

		if combineMode == 'function':
			self.combineFunction = torch.nn.Sequential(torch.nn.Linear(self.embeddingSize*3, self.embeddingSize), torch.nn.Tanh())
			self.combineEmbeddings = lambda embeddings : self.combineFunction(torch.cat(embeddings, dim=2))
		else:
			self.embeddingWeights = torch.nn.Parameter(torch.randn(3, 1)) \
									if combineMode == 'wsum' \
									else torch.randn(3, 1)
			self.combineEmbeddings = lambda embeddings : self.weightedSum(*embeddings)

		if activation == 'tanh':
			self.activation = torch.nn.Tanh()
		elif activation == 'relu':
			self.activation = torch.nn.ReLU()
		else:
			self.activation = torch.nn.Sigmoid()

		self.lstm = torch.nn.LSTM(input_size=self.embeddingSize,
								  hidden_size=self.hiddenEmbeddingSize,
								  num_layers=self.numLayers,
								  batch_first=True,
								  bidirectional=self.bidirectional)
		
		self.linearClassifier = torch.nn.Sequential()
		for i in range(len(self.hiddenSizes)-1):
			self.linearClassifier.add_module(f'linear{i}', torch.nn.Linear(self.hiddenSizes[i], self.hiddenSizes[i+1]))
			if i < len(self.hiddenSizes) - 1:
				self.linearClassifier.add_module(f'activation{i}', self.activation)

	def weightedSum(self, e1 : torch.Tensor, e2 : torch.Tensor, e3 : torch.Tensor) -> torch.Tensor:
		return e1 * self.embeddingWeights[0] + e2 * self.embeddingWeights[1] + e3 * self.embeddingWeights[2]

	def forward(self, X : torch.Tensor) -> torch.Tensor:
		# get the embeddings
		e1 = self.ELMo.embeddings(X)

		# get the output of the first LSTM
		e2, _ = self.ELMo.biLM1(e1)

		# get the output of the second LSTM
		e3, _ = self.ELMo.biLM2(e2)

		# combine the embeddings
		return self.combineEmbeddings((e1, e2, e3))

	def classificationForward(self, X : torch.Tensor) -> torch.Tensor:
		lengths = torch.sum(X != self.vocabulary['<PAD>'], dim=1).view(-1).to(torch.device('cpu'))

		# get embeddings from ELMo
		embeddings = self.forward(X)

		# pack padded sequence
		embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

		# get output of LSTM
		output, _ = self.lstm(embeddings)

		# select the last hidden state
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		output = torch.cat((output[torch.arange(output.shape[0]), lengths-1, :self.hiddenEmbeddingSize],
							output[torch.arange(output.shape[0]), 0, self.hiddenEmbeddingSize:]), dim=1)
		# output = output[:, -1, :]

		# get output of linear classifier
		output = self.linearClassifier(output)

		return output

	def preTrainElmo(self, batchSize: int = 16, learningRate: float = 0.001, epochs: int = 10, verbose: bool = True, retrain: bool = False, resumeFromCheckpoint: bool = False):
		self.ELMo.preTrain(batchSize=batchSize, learningRate=learningRate, epochs=epochs, verbose=verbose, retrain=retrain, resumeFromCheckpoint=resumeFromCheckpoint)

	def train(self, batchSize : int = 16, learningRate : float = 0.001, epochs : int = 10, verbose : bool = True, retrain : bool = False, resumeFromCheckpoint : bool = False) -> None:
		if not retrain and self.__loadClassifierModel__():
			if verbose:
				print('Model checkpoint loaded.')
			if not resumeFromCheckpoint:
				return
			else:
				if verbose:
					print('Resuming from checkpoint.')
		else:
			if verbose:
				print('Model checkpoint not found or retrain flag is set. Training from scratch.')

		if verbose:
			print("Training Classifier on sentence classification task.")

		trainLoader = DataLoader(self.trainDataset, batchSize, shuffle=True, collate_fn=self.trainDataset._custom_collate_)
		
		params = [ { 'params': self.linearClassifier.parameters() },
				   { 'params': self.lstm.parameters() } ]
		if self.combineMode == 'function':
			params.append({ 'params': self.combineFunction.parameters() })
		elif self.combineMode == 'wsum':
			params.append({ 'params': self.embeddingWeights })
		optimizer = torch.optim.Adam(params, lr=learningRate)
		
		criterion = torch.nn.CrossEntropyLoss()

		epochMetrics = {}
		self.to(self.device)
		if self.combineMode != 'function':
			self.embeddingWeights = self.embeddingWeights.to(self.device)
		
		for epoch in range(epochs):
			runningLoss = 0
			epochMetrics[epoch] = {}
			for X, y in tqdm(trainLoader, desc='Classifier Training', leave=True):
				X = X.to(self.device)
				y = y.to(self.device)

				optimizer.zero_grad()

				output = self.classificationForward(X)
				loss = criterion(output, y)
				loss.backward()

				optimizer.step()

				runningLoss += loss.item()
			
			epochMetrics[epoch]['train loss'] = runningLoss / len(trainLoader)

			if verbose:
				print(f'Epoch {epoch+1}/{epochs} | Loss: {epochMetrics[epoch]["train loss"]:.4f}')
			
			if (epoch+1) % 5 == 0:
				self.__saveClassifierModel__()
				if verbose:
					print(f'Epoch {epoch+1}/{epochs} | Model saved.')

		self.__saveClassifierModel__()
		if verbose:
			print('Classifier training complete. Model saved.\n')

	def evaluate(self, testDataset : NewsClassificationDataset) -> dict:
		"""
		Returns:
			metrics: dict of metrics like accuracy, precision, recall, f1, classification report
		"""
		metrics = {}
		
		preds = torch.tensor([]).to(self.device)
		testLoader = DataLoader(testDataset, 64, shuffle=False, collate_fn=testDataset._custom_collate_)
		
		self.to(self.device)
		if self.combineMode != 'function':
			self.embeddingWeights = self.embeddingWeights.to(self.device)
		with torch.no_grad():
			for X, y in tqdm(testLoader, desc='Classifier Evaluation', leave=True):
				X = X.to(self.device)
				y = y.to(self.device)

				output = self.classificationForward(X)

				predictions = torch.argmax(output, dim=1).view(-1)
				preds = torch.cat((preds, predictions))
		
		# copy to cpu
		preds = preds.cpu()

		metrics['accuracy'] = accuracy_score(testDataset.labels, preds)
		metrics['precision'] = precision_score(testDataset.labels, preds, average='macro', zero_division=0)
		metrics['recall'] = recall_score(testDataset.labels, preds, average='macro', zero_division=0)
		metrics['f1'] = f1_score(testDataset.labels, preds, average='macro', zero_division=0)
		metrics['report'] = classification_report(testDataset.labels, preds, zero_division=0)

		return metrics

	def __saveClassifierModel__(self) -> None:
		"""
		Save the current parameters of the model.
		"""
		if not os.path.exists(MODEL_CHECKPOINTS_PATH):
			os.mkdir(MODEL_CHECKPOINTS_PATH)

		if not os.path.exists(self._modelSavePath):
			os.mkdir(self._modelSavePath)
		
		torch.save(self.state_dict(), os.path.join(self._modelSavePath, f'classifier_model_{self._classifierModelFileSuffix}.pt'))

	def __loadClassifierModel__(self) -> bool:
		if not os.path.exists(os.path.join(self._modelSavePath, f'classifier_model_{self._classifierModelFileSuffix}.pt')):
			return False
		
		self.load_state_dict(torch.load(os.path.join(self._modelSavePath, f'classifier_model_{self._classifierModelFileSuffix}.pt')))
		return True

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(type(device), device)

	trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')

	elmoClassifier = ELMoClassifier(256, 						# embedding size
								 	trainDataset,				# train dataset
									device,						# device
									combineMode='function',
									hiddenEmbeddingSize=256,
									numLayers=2,
									hiddenSizes=[128, 64],
									activation='tanh')
	 
	elmoClassifier.preTrainElmo(batchSize=8, 
						 		learningRate=0.001, 
								epochs=10,
								retrain=False)

	elmoClassifier.train(batchSize=32,
					  	 epochs=10,
						 retrain=True,
						 learningRate=0.005)
	
	metrics = elmoClassifier.evaluate(trainDataset)
	print("Train Report:")
	print(metrics['report'])

	testDataset = NewsClassificationDataset('../data/News Classification Dataset/test.csv', trueVocab=trainDataset.vocabulary)
	testMetrics = elmoClassifier.evaluate(testDataset)
	
	print("\nTest Report:")
	print(testMetrics['report'])

	print(elmoClassifier.ELMo.evaluate(testDataset))