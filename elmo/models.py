import torch
import os
from torch.utils.data import DataLoader
from bidict import bidict
from typing import Literal
from tqdm import tqdm

from dataset import NewsClassificationDataset

MODEL_CHECKPOINTS_PATH = './model_checkpoints'

class ELMo(torch.nn.Module):
	"""
	ELMo model to obtain contextualized word embeddings.
	"""
	def __init__(self, embeddingSize : int, 
			  	 trainDataset : NewsClassificationDataset,
			  	 combineMode : Literal['function', 'wsum', 'sum'], 
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
		super(ELMo, self).__init__()


		self.__modelSavePath = os.path.join(MODEL_CHECKPOINTS_PATH, 'ELMo')
		self._modelFileSuffix = f'_{embeddingSize}_{combineMode}_{len(trainDataset.vocabulary)}'

		self.trainDataset = trainDataset
		self.vocabulary = trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)
		self.embeddingSize = embeddingSize
		self.device = device
		self.combineMode = combineMode
		self.combineEmbeddings = None # takes a tuple of 3 embeddings and returns the final embedding

		if combineMode == 'function':
			self.combineFunction = torch.nn.Sequential(torch.nn.Linear(embeddingSize*3, embeddingSize), torch.nn.Tanh())
			self.combineEmbeddings = lambda embeddings : self.combineFunction(torch.cat(embeddings, dim=2))
		else:
			self.embeddingWeights = torch.nn.Parameter(torch.randn(3, 1)) \
									if combineMode == 'wsum' \
									else torch.randn(3, 1)
			self.combineEmbeddings = lambda embeddings : self.weightedSum(*embeddings)

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

	def forward(self, input : torch.Tensor) -> torch.Tensor:
		# get the embeddings
		e1 = self.embeddings(input)

		# get the output of the first LSTM
		e2, _ = self.biLM1(e1)

		# get the output of the second LSTM
		e3, _ = self.biLM2(e2)
		# e3, _ = self.biLSTM2((e2+e1) / 2) # residual connection

		# combine the embeddings
		return self.combineEmbeddings((e1, e2, e3))

	def nextWordPredictionForward(self, X : torch.Tensor) -> torch.Tensor:
		# get the embeddings
		e1 = self.embeddings(X)
		
		# get the output of the first LSTM
		e2, _ = self.biLM1(e1)

		# get the output of the second LSTM
		e3, _ = self.biLM2(e2)

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

	def weightedSum(self, e1 : torch.Tensor, e2 : torch.Tensor, e3 : torch.Tensor) -> torch.Tensor:
		return e1 * self.embeddingWeights[0] + e2 * self.embeddingWeights[1] + e3 * self.embeddingWeights[2]

	def preTrain(self, 
			  	 batchSize : int = 16, learningRate : float = 0.001, 
				 epochs : int = 10, verbose : bool = True, 
				 retrain : bool = False) -> None:
		if not retrain and self.__loadModel__():
			if verbose:
				print('Model checkpoint loaded.')
			return
		else:
			if verbose:
				print('Model checkpoint not found or retrain flag is set. Training from scratch.')

		if verbose:
			print("Commencing pretraining...")
			
		trainLoader = DataLoader(self.trainDataset, batchSize, shuffle=True, collate_fn=self.trainDataset._custom_collate_)
		optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
		criterion = torch.nn.CrossEntropyLoss(ignore_index=len(self.trainDataset.vocabulary)-1)

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

	def __saveModel__(self) -> None:
		"""
		Save the current parameters of the model.
		"""
		if not os.path.exists(MODEL_CHECKPOINTS_PATH):
			os.mkdir(MODEL_CHECKPOINTS_PATH)

		if not os.path.exists(self.__modelSavePath):
			os.mkdir(self.__modelSavePath)
		
		torch.save(self.state_dict(), os.path.join(self.__modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt'))

	def __loadModel__(self) -> bool:
		if not os.path.exists(os.path.join(self.__modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt')):
			return False
		
		self.load_state_dict(torch.load(os.path.join(self.__modelSavePath, f'pretrained_model_{self._modelFileSuffix}.pt')))
		return True

class ELMoClassifier(ELMo):
	def __init__(self, *elmoParams,
			  	 hiddenEmbeddingSize : int = 256, 
				 numLayers : int = 1,
			  	 hiddenSizes : list[int] = [64],
				 activation : Literal['tanh', 'relu', 'sigmoid'] = 'tanh') -> None:
		super().__init__(*elmoParams)

		self.__modelSavePath = os.path.join(MODEL_CHECKPOINTS_PATH, 'ELMo_classifier')
		self._classifierModelFileSuffix = self._modelFileSuffix + f"_{hiddenEmbeddingSize}_{numLayers}_{hiddenSizes}_{activation}"

		self.bidirectional = True
		self.trainDataset = trainDataset
		self.vocabulary = trainDataset.vocabulary
		self.vocabSize = len(self.vocabulary)
		self.classes = trainDataset.classes
		self.numClasses = trainDataset.numClasses
		self.hiddenSizes = [ hiddenEmbeddingSize*(self.bidirectional+1) ] + hiddenSizes + [ self.numClasses ]

		if activation == 'tanh':
			self.activation = torch.nn.Tanh()
		elif activation == 'relu':
			self.activation = torch.nn.ReLU()
		else:
			self.activation = torch.nn.Sigmoid()

		self.lstm = torch.nn.LSTM(input_size=self.embeddingSize,
								  hidden_size=hiddenEmbeddingSize,
								  num_layers=numLayers,
								  batch_first=True,
								  bidirectional=self.bidirectional)
		
		self.linearClassifier = torch.nn.Sequential()
		for i in range(len(self.hiddenSizes)-1):
			self.linearClassifier.add_module(f'linear{i}', torch.nn.Linear(self.hiddenSizes[i], self.hiddenSizes[i+1]))
			if i < len(self.hiddenSizes) - 1:
				self.linearClassifier.add_module(f'activation{i}', self.activation)
	
	def classificationForward(self, X : torch.Tensor) -> torch.Tensor:
		# get embeddings from ELMo
		embeddings = self.forward(X)

		# get output of LSTM
		output, _ = self.lstm(embeddings)

		# select the last hidden state
		output = output[:, -1, :]

		# get output of linear classifier
		return self.linearClassifier(output)

	def train(self, 
			  batchSize : int = 16, learningRate : float = 0.001, 
			  epochs : int = 10, verbose : bool = True, 
			  retrain : bool = False) -> None:
		if not retrain and self.__loadClassifierModel__():
			if verbose:
				print('Model checkpoint loaded.')
			return
		else:
			if verbose:
				print('Model checkpoint not found or retrain flag is set. Training from scratch.')

		if verbose:
			print("Training Classifier on sentence classification task...")

		trainLoader = DataLoader(self.trainDataset, batchSize, shuffle=True, collate_fn=self.trainDataset._custom_collate_)
		
		params = [ { 'params': self.linearClassifier.parameters() },
				   { 'params': self.lstm.parameters() } ]
		if self.combineMode == 'function':
			params.append({ 'params': self.combineFunction.parameters() })
		elif self.combineMode == 'wsum':
			params.append({ 'params': self.embeddingWeights })
		
		epochMetrics = {}

		optimizer = torch.optim.Adam(params, lr=learningRate)
		criterion = torch.nn.CrossEntropyLoss()

		self.to(self.device)
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
			
			if (epoch+1) % 2 == 0:
				self.__saveClassifierModel__()
				if verbose:
					print(f'Epoch {epoch+1}/{epochs} | Model saved.')
	
	def evaluate(self, testDataset : NewsClassificationDataset) -> dict:
		"""
		Returns:
			metrics: dict of metrics like accuracy, precision, recall, f1, classification report
		"""

	def __saveClassifierModel__(self) -> None:
		"""
		Save the current parameters of the model.
		"""
		if not os.path.exists(MODEL_CHECKPOINTS_PATH):
			os.mkdir(MODEL_CHECKPOINTS_PATH)

		if not os.path.exists(self.__modelSavePath):
			os.mkdir(self.__modelSavePath)
		
		torch.save(self.state_dict(), os.path.join(self.__modelSavePath, f'classifier_model_{self._classifierModelFileSuffix}.pt'))

	def __loadClassifierModel__(self) -> bool:
		if not os.path.exists(os.path.join(self.__modelSavePath, f'classifier_model_{self._classifierModelFileSuffix}.pt')):
			return False
		
		self.load_state_dict(torch.load(os.path.join(self.__modelSavePath, f'classifier_model_{self._classifierModelFileSuffix}.pt')))
		return True\

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(type(device), device)

	trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')

	elmoClassifier = ELMoClassifier(256,
								 	trainDataset,
									'function',
									device)
	
	elmoClassifier.preTrain(batchSize=8, 
						 	learningRate=0.001, 
							epochs=10, 
							retrain=True)
	elmoClassifier.train(batchSize=32,
					  	 epochs=20)