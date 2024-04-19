from elmo.models import ELMoClassifier
from elmo.dataset import NewsClassificationDataset

import optuna
import torch

trainDataset = NewsClassificationDataset('../data/News Classification Dataset/train.csv')
testDataset = NewsClassificationDataset('../data/News Classification Dataset/test.csv', trueVocab=trainDataset.vocabulary)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial : optuna.Trial):
	combineMode = trial.suggest_categorical('combineMode', ['function', 'wsum', sum])
	activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
	epochs = trial.suggest_int(name='epochs', low=5, high=15, step=5)
	# embeddingSize = trial.suggest_int(name='embeddingSize', low=128, high=512, step=128)
	# hiddenEmbeddingSize = trial.suggest_int(name='hiddenEmbeddingSize', low=128, high=512, step=128)

	embeddingSize = 256
	hiddenEmbeddingSize = 128
	elmoClassifier = ELMoClassifier(embeddingSize,
								 	trainDataset,
									device,
									combineMode=combineMode,
									hiddenEmbeddingSize=hiddenEmbeddingSize,
									numLayers=2,
									hiddenSizes=[128, 64],
									activation=activation)
	
	elmoClassifier.preTrainElmo(batchSize=8,
								learningRate=0.001,
								epochs=10,
								retrain=False)
	
	elmoClassifier.train(batchSize=32,
					     epochs=epochs,
						 retrain=True,
						 learningRate=0.005)
	
	metrics = elmoClassifier.evaluate(testDataset)

	return metrics['f1']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

prunedTrials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
completeTrials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

print("Study statistics: ")
print("\tNumber of finished trials: ", len(study.trials))
print("\tNumber of pruned trials: ", len(prunedTrials))
print("\tNumber of complete trials: ", len(completeTrials))

print("\tBest trial:")
trial = study.best_trial

print("\tBest F1 Value: ", trial.value)

print("\tBest hyperparams: ")
for key, value in trial.params.items():
	print("\t{}: {}".format(key, value))


	
