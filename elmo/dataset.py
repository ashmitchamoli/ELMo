import numpy as np
import torch
import time
from bidict import bidict
from torch.utils.data import Dataset

from preprocessor import DatasetPreprocessor

class NewsClassificationDataset(Dataset):
	def __init__(self, datasetFilePath : str, trueVocab : bidict = None) -> None:
		super().__init__()
	
		self.datasetFilePath = datasetFilePath
		self.preprocessor = DatasetPreprocessor(datasetFilePath)

		print('Processing dataset...')
		startTime = time.time()
		self.tokens, self.labels, self.vocabulary = self.preprocessor.processFile(trueVocab)
		print(f'Processed dataset in {time.time() - startTime} seconds.')
		self.classes = np.unique(self.labels)
		self.numClasses = len(self.classes)

	def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Returns: X, y
			X : sentence, a tensor of word indices according to self.vocabulary
			y : label, a tensor of classes
		"""
		return self.tokens[index], self.labels[index]
	
	def __len__(self) -> int:
		return len(self.tokens)

	def _custom_collate_(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Returns: X, y
			X : sentence, a tensor of word indices according to self.vocabulary
			y : label, a tensor of classes
		"""
		tokens, labels = zip(*batch)

		tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.vocabulary['<PAD>'])
		labels = torch.stack(labels)
		
		return tokens, labels