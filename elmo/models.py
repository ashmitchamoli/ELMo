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

		if combineMode == 'function':
			self.combineEmbeddings = torch.nn.Linear(embeddingSize*3, embeddingSize)
		else:
			self.embeddingWeights = torch.nn.Parameter(torch.randn(3, 1)) \
									if combineMode == 'wsum' \
									else torch.randn(3, 1) 
			self.combineEmbeddings = lambda embeddings : self.weightedSum(*embeddings)
		

		self.embeddings = torch.nn.Embedding(self.vocabSize, self.embeddingSize)
		self.biLSTM1 = torch.nn.LSTM(input_size=embeddingSize,
							   		 hidden_size=embeddingSize,
									 num_layers=1,
									 bidirectional=True,
									 batch_first=True)
		self.biLSTM2 = torch.nn.LSTM(input_size=embeddingSize*2,
							   		 hidden_size=embeddingSize,
									 num_layers=1,
									 bidirectional=True,
									 batch_first=True)
	
	def forward(self, input : torch.Tensor) -> torch.Tensor:
		
		pass

	def weightedSum(self, e1 : torch.Tensor, e2 : torch.Tensor, e3 : torch.Tensor) -> torch.Tensor:
		return e1 * self.embeddingWeights[0] + e2 * self.embeddingWeights[1] + e3 * self.embeddingWeights[2]