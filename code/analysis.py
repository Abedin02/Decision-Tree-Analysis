from DecisionTree import DecisionTree
from utils import plot_decision_boundary, load_random_dataset, load_circle_dataset, plot_dataset
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, dirname, abspath, exists
from os import mkdir
 
 ########################
# Create plots directory if it doesn't exist
########################

plots_directory = join(dirname(abspath(__file__)), '..', 'plots')
if not exists(plots_directory):
	mkdir(plots_directory)

########################
########################

def part1():

	"""
	a decision tree model is trained with unconstrained depth on a random
	dataset. Create a decision boundary plot, printing the accuracy to the console.
	Save plot to the plots folder.

	"""
	
	
	# load dataset
	X, y = load_random_dataset()

	dtree = DecisionTree()
	dtree.train(X, y)
	
	# create empty figure
	fig = plt.figure()

	fig = plot_decision_boundary(X, y, dtree, 'Decision Boundary with Unconstrained Depth')
	
	
	# instantiate accuracy
	accuracy = DecisionTree.accuracy_score(dtree, X, y)


	# save figure and print accuracy
	print('Part 1 Accuracy:', accuracy)
	fig.savefig(join(plots_directory, 'part1.png'))


def part2():

	"""
	A decision tree model is created with unconstrained depth on a circle training 
	dataset. decision boundary is plotted overlayed both on the training
	and validation datasets. Print to he console the accuracy scores for both the training
	and validation datasets.

	"""

	# load dataset
	X_train, y_train, X_val, y_val = load_circle_dataset()

	dtree = DecisionTree()
	dtree.train(X_train, y_train)

	# create empty figure
	fig_training, fig_validation = plt.figure(), plt.figure()
	fig_training = plot_decision_boundary(X_train, y_train, dtree, 'Decision Boundary on Training Set')
	fig_validation = plot_decision_boundary(X_val, y_val, dtree, 'Decision Boundary on Validation Set')
	# instantiate accuracy
	training_accuracy, validation_accuracy = None, None
	training_accuracy = DecisionTree.accuracy_score(dtree, X_train, y_train)
	validation_accuracy = DecisionTree.accuracy_score(dtree, X_val, y_val)
	

	# save figure and print accuracy
	print('Part 2 Training Accuracy:', training_accuracy)
	print('Part 2 Validation Accuracy:', validation_accuracy)
	fig_training.savefig(join(plots_directory, 'part2_training.png'))
	fig_validation.savefig(join(plots_directory, 'part2_validation.png'))


def part3():

	"""
	A decision tree is trained with max_depth=1 on the circle training 
	dataset. decision boundary plots are created overlayed both over the training
	and validation datasets. Print to the console the accuracy scores for both the training
	and validation datasets.

	"""

	# load dataset
	X_train, y_train, X_val, y_val = load_circle_dataset()

	dtree = DecisionTree(max_depth=1)
	dtree.train(X_train, y_train)
	# create empty figure
	fig_training, fig_validation = plt.figure(), plt.figure()
	fig_training = plot_decision_boundary(X_train, y_train, dtree, 'Decision Boundary on Training Set with max depth=1')
	fig_validation = plot_decision_boundary(X_val, y_val, dtree, 'Decision Boundary on Validation Set with max depth=1')
	# instantiate accuracy
	training_accuracy, validation_accuracy = None, None
	training_accuracy = DecisionTree.accuracy_score(dtree, X_train, y_train)
	validation_accuracy = DecisionTree.accuracy_score(dtree, X_val, y_val)
	


	# save figure and print accuracy
	print('Part 3 Training Accuracy:', training_accuracy)
	print('Part 3 Validation Accuracy:', validation_accuracy)
	fig_training.savefig(join(plots_directory, 'part3_training.png'))
	fig_validation.savefig(join(plots_directory, 'part3_validation.png'))




def part4():

	"""
	A hyperparameter tuning curve is created by training multiple decision tree models, varying
	the max_depth hyerparmeter from 1 to 20, and making line plots for both the training 
	and validation accuracies. Print to the console the optimal max depth.

	"""

	# load circle dataset
	X_train, y_train, X_val, y_val = load_circle_dataset()

	# instantiate figure
	fig, ax = plt.subplots()
	fig.suptitle('Hyperparameter Tuning Curve')
	ax.set_xlabel('Max Depth')
	ax.set_ylabel('Accuracy')
	# instantiate lists to store accuracies
	training_accuracies, validation_accuracies = [], []
	for max_depth in range(1, 21):
		dtree = DecisionTree(max_depth=max_depth)
		dtree.train(X_train, y_train)
		training_accuracy = DecisionTree.accuracy_score(dtree, X_train, y_train)
		validation_accuracy = DecisionTree.accuracy_score(dtree, X_val, y_val)
		training_accuracies.append(training_accuracy)
		validation_accuracies.append(validation_accuracy)
	ax.set_yticks(np.arange(0, training_accuracy + 0.1, 0.1))
	ax.set_xticks(np.arange(1, 21, 1))
	ax.plot(range(1, 21), training_accuracies, label='Training Accuracy')
	ax.plot(range(1, 21), validation_accuracies, label='Validation Accuracy')
	ax.legend()


	# initialize optimal depth
	optimal_max_depth = None
	optimal_max_depth = np.argmax(validation_accuracies) + 1
	
	

	# print optimal max depth and save
	print('Part 4 Optimal Max Depth:', optimal_max_depth)
	fig.savefig(join(plots_directory, 'part4_hyperparemeter_tuning_plot.png'))
	
	

def part5():

	"""
	For the optimal max depth found from part 4, decision boundary plots are created 
	overlayed both over the training and validation datasets.
	The accuracy scores for both the training and validation datasets are printed to the console.
	
	"""

	# load circle dataset
	X_train, y_train, X_val, y_val = load_circle_dataset()
	optimal_max_depth = 4
	dtree = DecisionTree(max_depth=optimal_max_depth)
	dtree.train(X_train, y_train)

	# create empty figure
	fig_training, fig_validation = plt.figure(), plt.figure()

	fig_training = plot_decision_boundary(X_train, y_train, dtree, f'Decision Boundary on Training Set with Optimal Max Depth={optimal_max_depth}')
	fig_validation = plot_decision_boundary(X_val, y_val, dtree, f'Decision Boundary on Validation Set with Optimal Max Depth={optimal_max_depth}')
	# instantiate accuracy
	training_accuracy, validation_accuracy = None, None
	training_accuracy = DecisionTree.accuracy_score(dtree, X_train, y_train)
	validation_accuracy = DecisionTree.accuracy_score(dtree, X_val, y_val)


	# save figure and print accuracy
	print('Part 5 Training Accuracy:', training_accuracy)
	print('Part 5 Validation Accuracy:', validation_accuracy)
	fig_training.savefig(join(plots_directory, 'part5_training.png'))
	fig_validation.savefig(join(plots_directory, 'part5_validation.png'))
	


if __name__ == '__main__':
	part1()
	part2()
	part3()
	part4()
	part5()

