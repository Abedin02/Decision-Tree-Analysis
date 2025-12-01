from utils import entropy, information_gain, best_split_for_feature, \
					best_feature_to_split_on, load_random_dataset, load_circle_dataset
from DecisionTree import DecisionTree
import numpy as np


import sys

print(sys.prefix)
def test_entropy():

	"""
	tests entropy function

	"""

	print('------------')
	print('Test Entropy')
	print('------------')

	test_cases = [
		[np.array([2, 2, 2, 2]), 0],
		[np.array([1, 1, 2, 2]), 1],
		[np.array([2, 1, 0, 1, 2, 1, 3]), 1.842],
		[np.array([3]), 0]
	]

	for test_number, (k, v) in enumerate(test_cases, start=1):

		result = entropy(k)

		if result is None:
			print(f'Case {test_number}: FAILED - output is None')
			continue

		# round
		result = np.round(result, 3)

		if v != result:
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} but got {result}')
		else:
			print(f'Case {test_number}: passed')

	print()


def test_information_gain():

	"""
	tests entropy function

	"""

	print('---------------------')
	print('Test Information Gain')
	print('---------------------')


	test_cases = [
		[[np.array([0,0,0,1,1,1]), np.array([0,0,0]), np.array([1,1,1])],1],
		[[np.array([0,0,0,1,1]), np.array([0,0,0]), np.array([1,1])],.971],
		[[np.array([1,0,1,1,1]), np.array([1,0,1]), np.array([1,1])],.171],
		[[np.array([1,2,1,3,1]), np.array([1,2]), np.array([1,3,1])], .42]
	]

	for test_number, (k, v) in enumerate(test_cases, start=1):

		result = information_gain(*k)

		if result is None:
			print(f'Case {test_number}: FAILED - output is None')
			continue


		result = np.round(result, 3)

		if v != result:
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} but got {result}')
		else:
			print(f'Case {test_number}: passed')

	print()


def test_best_split_for_feature():

	"""
	tests best split for feature function

	"""

	print('---------------------------')
	print('Test Best Split for Feature')
	print('---------------------------')


	test_cases = [
		[[np.array([5, 1, 6, 12, 1]), np.array([1, 0, 0, 1, 1])],[9,.171]],
		[[np.array([-4, 1, 3, 2, 9]), np.array([1,2,2,1,0])],[6,.722]],
	]

	for test_number, (k, v) in enumerate(test_cases, start=1):
		split_val, in_gain = best_split_for_feature(*k)

		in_gain = np.round(in_gain,3)
		if split_val is None or in_gain is None:
			print(f'Case {test_number}: FAILED - output is None')
			continue

		#print("testing split val and in gain", np.round(split_val) , in_gain)	
		if v != [np.round(split_val,3), np.round(in_gain,3)] :
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} but got {[split_val, in_gain]}')
		else:
			print(f'Case {test_number}: passed')

	print()


def test_best_feature_to_split_on():

	"""
	tests for best feature to split on

	"""

	print('-----------------------------')
	print('Test Best Feature to Split On')
	print('-----------------------------')

	X, y= load_random_dataset()
	
	X_c, y_c, _, _ = load_circle_dataset()
	test_cases = [
		[[X, y],[1,.3, 7.349]],
		[[X_c, y_c],[1,.163, 2.685]],
	]

	for test_number, (k, v) in enumerate(test_cases, start=1):
		idx, in_gain, split_val = best_feature_to_split_on(*k)

		if idx is None or in_gain is None or split_val is None:
			print(f'Case {test_number}: FAILED - output is None')
			continue

		if v != [idx, np.round(in_gain,3), np.round(split_val,3)] :
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} but got {[idx, in_gain, split_val]}')
		else:
			print(f'Case {test_number}: passed')

	print()


def test_decision_tree_train():

	"""
	tests for decision tree train

	"""

	print('-------------------------')
	print('Test Decision Tree Train')
	print('-------------------------')

	# load random dataset
	X, y= load_random_dataset()

	test_cases = []

	# test decision tree with depth of 1 
	dtree = DecisionTree(max_depth=1)
	dtree.train(X, y)
	split_vals = dtree.in_order_split_vals()
	test_cases.append(
		[split_vals, [None, 7.349, None]]
	)
	
	# test decision tree with max depth of 2
	dtree = DecisionTree(max_depth=2)
	dtree.train(X, y)
	split_vals = dtree.in_order_split_vals()
	test_cases.append(
		[split_vals,  [None, 1.495, None, 7.349, None]]
	)

	# test decision tree with unlimited depth
	dtree = DecisionTree(max_depth=None)
	dtree.train(X, y)
	split_vals = dtree.in_order_split_vals()
	test_cases.append(
		[split_vals, [None, 4.42, None, 1.495, None, 2.153, None, 2.668, None, 3.074, None, 3.794, None, 6.662, None, 3.503, None, 6.425, None, 6.909, None, 7.673, None, 8.534, None, 9.301, None, 7.349, None]]
	)

	
	for test_number, (k, v) in enumerate(test_cases, start=1):
		if k != v:
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} split vals but got {k}')
		else:
			print(f'Case {test_number}: passed')

	print()


def test_decision_tree_predict():

	"""
	tests for decision tree predict

	"""

	print('-------------------------')
	print('Test Decision Tree Predict')
	print('-------------------------')

	# load random dataset
	X, y= load_random_dataset()

	test_cases = []

	# test decision tree with depth of 1 
	dtree = DecisionTree(max_depth=1)
	dtree.train(X, y)
	predictions = dtree.predict(X)
	
	test_cases.append(
		[predictions.tolist(), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1]]
	)
	
	# test decision tree with max depth of 2
	dtree = DecisionTree(max_depth=2)
	dtree.train(X, y)
	predictions = dtree.predict(X)
	test_cases.append(
		[predictions.tolist(), [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1]]
	)

	# test decision tree with unlimited depth
	dtree = DecisionTree(max_depth=None)
	dtree.train(X, y)
	predictions = dtree.predict(X)
	test_cases.append(
		[predictions.tolist(), [1, 2, 0, 0, 2, 1, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 2, 2, 1, 0, 1, 1]]
	)
	
	for test_number, (k, v) in enumerate(test_cases, start=1):
		if k != v:
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} predictions but got {k}')
		else:
			print(f'Case {test_number}: passed')

	print()


def test_decision_tree_accuracy():

	"""
	tests for decision tree accuracy

	"""

	print('---------------------------')
	print('Test Decision Tree Accuracy')
	print('---------------------------')

	# load random dataset
	X, y= load_random_dataset()

	test_cases = []

	# test decision tree with depth of 1 
	dtree = DecisionTree(max_depth=1)
	dtree.train(X, y)
	acc = np.round(dtree.accuracy_score(X, y),3)
	test_cases.append(
		[acc,  .633]
	)
	
	# test decision tree with max depth of 2
	dtree = DecisionTree(max_depth=2)
	dtree.train(X, y)
	acc = np.round(dtree.accuracy_score(X, y), 3)
	test_cases.append(
		[acc,  .7]
	)
	
	# test decision tree with unlimited depth
	dtree = DecisionTree(max_depth=None)
	dtree.train(X, y)
	acc = np.round(dtree.accuracy_score(X, y), 3)
	test_cases.append(
		[acc,  1]
	)
	
	
	for test_number, (k, v) in enumerate(test_cases, start=1):
		if k != v:
			print(f'Case {test_number}: FAILED')
			print(f'Expected {v} accuracy but got {k}')
		else:
			print(f'Case {test_number}: passed')

	print()
	
	


def run_tests():

	"""
	runs all tests

	"""

	print('')
	print('#############')
	print('Run All Tests')
	print('#############')
	print('')

	test_entropy()
	test_information_gain()
	test_best_split_for_feature()
	test_best_feature_to_split_on()
	test_decision_tree_train()
	test_decision_tree_predict()
	test_decision_tree_accuracy()



if __name__ == '__main__':

	run_tests()




