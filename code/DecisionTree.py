import numpy as np
from scipy import stats
from DecisionTreeNode import DecisionTreeNode
from utils import best_feature_to_split_on, entropy

class DecisionTree:

	"""
	Object to represent a decision tree model
	
	"""

	def __init__(self, max_depth=None):

		"""
		Constructor called when object is initialized. Only need
		to store max depth hyperparemeter and initialize root node to None.
		The only hyperparameter is max depth. When it is None this should
		train a decision tree with unconstrained depth.

		"""

		# store hyperparameters
		self.max_depth = max_depth

		# root node
		self.root_node = None


	def train(self, X, y):

		"""
		trains a decision tree by calling internal _train recursive method

		"""

		self.root_node = self._train(X, y, depth=0)


	def _train(self, X, y, depth):
		all_same = np.all(y == y[0])
		if all_same == True:
			return DecisionTreeNode(is_leaf=True, label=y[0])
		
		if depth == self.max_depth:
			self.is_leaf = True
			self.label = stats.mode(y).mode
			return DecisionTreeNode(is_leaf=True, label=self.label)
		
		split_idx, best_info_gain, split_val = best_feature_to_split_on(X, y)
		
		left_mask = X[:, split_idx] < split_val
		right_mask = X[:, split_idx] >= split_val
		left_X = X[left_mask]
		left_y = y[left_mask]
		right_X = X[right_mask]
		right_y = y[right_mask]
		left_node = self._train(left_X, left_y, depth+1)
		right_node = self._train(right_X, right_y, depth+1)
		self.is_leaf = False
		self.label = None
		return DecisionTreeNode(is_leaf=False, split_idx=split_idx, split_val=split_val, entropy=entropy(y), left_node=left_node, right_node=right_node)
	

	def predict(self, X):

		"""
		computes predictions for multiple instances by iterating through
		each example and computing its prediction


		Args:
			X (2D numpy array): A numpy array of size (n x d) where each row a training instance

		Returns:
			y_pred: 1d array of predicted labels

		"""

		# list to store predictions
		y_pred = np.array([])

		# iterate through all examples to compute predictions and append to y_pred
		for x in X:
			pred = self._predict(self.root_node, x)

			y_pred = np.append(y_pred, pred)

		return y_pred


	def _predict(self, node, x):

		# Computes decision tree prediction for one example. This function is implemented
		# recursively.

		if node.is_leaf:
			return node.label
		split_idx = node.split_idx
		split_val = node.split_val
		if x[split_idx] < split_val:
			return self._predict(node.left_node, x)
		else:
			return self._predict(node.right_node, x)
		


		pass



	def accuracy_score(self, X, y):

		"""
		Compute the decision tree prediction of X and compute the accuracy compared to 
		the actual labels y

		"""
		pred = self.predict(X)
		matches = pred == y
		num_correct = np.sum(matches)

		accuracy = num_correct/len(y)

		
		return accuracy


	def visualize_tree(self):

		"""
		visualizes decision tree

		"""

	

		self._visualize_tree(self.root_node, level=0)

	def _visualize_tree(self, node, level):

		"""
		visualizes decision tree


		"""

		if node is not None:
			self._visualize_tree(node.right_node, level + 1)
			print(" " * (level * 30) + str(node))
			self._visualize_tree(node.left_node, level + 1)


	def in_order_split_vals(self):

		"""
		Performs in order walk on tree and reutns split values. Return None
		value if node is a leaf node.


		"""

		split_vals = []

		self._in_order_split_vals(self.root_node, split_vals=split_vals)

		return split_vals


	def _in_order_split_vals(self, node, split_vals):

		"""
		Performs in order walk on tree and reutns split values. Return None
		value if node is a leaf node.


		"""

		if node is not None:
			self._in_order_split_vals(node.left_node, split_vals)
			split_vals.append(np.round(node.split_val, 3) if not node.is_leaf else None)
			self._in_order_split_vals(node.right_node, split_vals)

