import nltk
import torch
import re
import os
import pandas as pd
import pickle as pkl
from bidict import bidict

DATA_CACHE_DIR = "./data_cache"

class DatasetPreprocessor:
	def __init__(self, datasetFilePath : str) -> None:
		self.stemmer = nltk.stem.PorterStemmer()
		self.datasetFilePath = datasetFilePath
		self.fileName = self.datasetFilePath.split('/')[-1][:-4]
		self._saveFileName = f"data_{self.fileName}.pkl"
	
	def processFile(self, trueVocab : bidict = None) -> tuple[list[list[str]], list[int], bidict]:
		"""
		Returns:
			tokens: a list of tensor of tokenIDs.
			labels: a tensor of labels corresponding to each sentence.
			vocabulary: bidict of word and index.
		"""
		if self.__loadProcessedData__():
			print(f'Loaded processed data from {self._saveFileName}.')

			if trueVocab is not None:
				print("Converting tokens to given vocab.")
				self.tokens = [ torch.tensor([ trueVocab.get(self.vocabulary.inverse[token.item()], trueVocab['<UNK>']) for token in sentence ]) for sentence in self.tokens ]
				self.vocabulary = trueVocab

			return self.tokens, self.labels, self.vocabulary
		
		df = pd.read_csv(self.datasetFilePath)
		vocabulary = set()
		tokens = []
		labels = []

		def processDf(row):
			# tokenize sentence
			sentenceTokens = ['<S>'] + self.tokenizeSentence(row['Description']) + ['</S>']
			
			# add sentence to tokens lsit
			tokens.append(sentenceTokens)
			
			# add tokens to labels list
			labels.append(row['Class Index'] - 1)

			# update vocabulary
			vocabulary.update(sentenceTokens)

			return row

		df.apply(processDf, axis=1)

		vocabulary.add('<UNK>')
		vocabulary.add('<PAD>')
		self.vocabulary = bidict({ word : index for index, word in enumerate(vocabulary) }) 
		self.tokens = [ torch.tensor([ self.vocabulary.get(token, self.vocabulary['<UNK>']) for token in sentence ]) for sentence in tokens ]
		self.labels = torch.tensor(labels)

		# save tokens, labels and vocabulary
		self.__saveProcessedData__()
		print(f"Processed data saved to {self._saveFileName}.")
		
		if trueVocab is not None:
			self.tokens = [ torch.tensor([ trueVocab.get(token, trueVocab['<UNK>']) for token in sentence ]) for sentence in tokens ]

		return self.tokens, self.labels, self.vocabulary if trueVocab is None else trueVocab

	def tokenizeSentence(self, sentence : str) -> list[str]:
		"""
		Returns:
			tokens: a list of tokens.
		"""
		rawTokens = nltk.tokenize.word_tokenize(sentence)

		tokens = []
		for token in rawTokens:
			words = re.split('[^a-zA-Z0-9]', token)
			words = [ self.stemmer.stem(word.strip().lower()) for word in words if word.strip() != '' ]
			tokens.extend(words)
		
		return tokens
	
	def __saveProcessedData__(self):
		if not os.path.exists(DATA_CACHE_DIR):
			os.mkdir(DATA_CACHE_DIR)
		
		pkl.dump((self.tokens, self.labels, self.vocabulary), 
		   		 open(os.path.join(DATA_CACHE_DIR, self._saveFileName), 'wb'))
	
	def __loadProcessedData__(self) -> bool:
		if not os.path.exists(os.path.join(DATA_CACHE_DIR, self._saveFileName)):
			return False
		
		self.tokens, self.labels, self.vocabulary = pkl.load(open(os.path.join(DATA_CACHE_DIR, self._saveFileName), 'rb'))
		
		return True