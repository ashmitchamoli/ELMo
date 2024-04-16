import nltk
import torch
import re
import pandas as pd
from bidict import bidict

class DatasetPreprocessor:
	def __init__(self, datasetFilePath : str) -> None:
		self.stemmer = nltk.stem.PorterStemmer()
		self.datasetFilePath = datasetFilePath
	
	def processFile(self) -> tuple[list[list[str]], list[int], bidict]:
		"""
		Returns:
			tokens: a list of tensor of tokenIDs.
			labels: a tensor of labels corresponding to each sentence.
			vocabulary: bidict of word and index.
		"""
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

		tokens = [ torch.tensor([ self.vocabulary[token] for token in sentence ]) for sentence in tokens ]
		labels = torch.tensor(labels)

		self.tokens = tokens
		self.labels = labels

		return tokens, labels, vocabulary

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