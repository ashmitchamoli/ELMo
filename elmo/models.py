import torch
from bidict import bidict
from typing import Literal

class ELMo(torch.nn.Module):
	"""
	ELMo model to obtain contextualized word embeddings.
	"""
	def __init__(self, embeddingSize : int, vocabulary : bidict, combineMode : Literal['function', 'wsum', 'sum']) -> None:
		"""
		Parameters:
			embeddingSize (int): The size of the embedding.
			vocabulary (bidict): A bidirectional dictionary containing the vocabulary and word indices.
			combineMode (Literal['function', 'wsum', 'sum']): The mode to combine embeddings.
				- 'function': A linear layer is used to combine embeddings.
				- 'wsum': Take the weighted sum of embeddings using trainable weights.
				- 'sum': Take the weighted sum of embeddings using fixed random weights.
		"""
		super().__init__()

		self.embeddingSize = embeddingSize
		self.vocabulary = vocabulary
		self.vocabSize = len(vocabulary)
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

		self.embeddings = torch.nn.Embedding(self.vocabSize, self.embeddingSize)
		self.biLM1 = torch.nn.LSTM(input_size=embeddingSize,
							   		 hidden_size=embeddingSize,
									 num_layers=1,
									 bidirectional=True,
									 batch_first=True,
									 merge_mode='ave')
		self.biLSTM2 = torch.nn.LSTM(input_size=embeddingSize,
							   		 hidden_size=embeddingSize,
									 num_layers=1,
									 bidirectional=True,
									 batch_first=True,
									 merge_mode='ave')

	def forward(self, input : torch.Tensor) -> torch.Tensor:
		# get the embeddings
		e1 = self.embeddings(input)

		# get the output of the first LSTM
		e2, _ = self.biLSTM1(e1)

		# get the output of the second LSTM
		e3, _ = self.biLSTM2(e2)
		# e3, _ = self.biLSTM2((e2+e1) / 2) # residual connection

		# combine the embeddings
		return self.combineEmbeddings((e1, e2, e3))

	def weightedSum(self, e1 : torch.Tensor, e2 : torch.Tensor, e3 : torch.Tensor) -> torch.Tensor:
		return e1 * self.embeddingWeights[0] + e2 * self.embeddingWeights[1] + e3 * self.embeddingWeights[2]

class ELMoClassifier(ELMo):
	def __init__(self, *elmoParams, hiddenSizes : list[int], activation : Literal['tanh', 'relu', 'sigmoid']) -> None:
		super().__init__(*elmoParams)

		self.hiddenSizes = hiddenSizes
		if activation == 'tanh':
			self.activation = torch.nn.Tanh()
		elif activation == 'relu':
			self.activation = torch.nn.ReLU()
		else:
			self.activation = torch.nn.Sigmoid()

		self.classifier = torch.nn.Sequential()
