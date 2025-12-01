import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from os.path import join
from os.path import dirname, abspath
"""
utility functions

"""


def entropy(S):
  
  unique_vals, counts = np.unique(S, return_counts=True)
  probabilities = counts / len(S)
  en = 0
  for p in probabilities:
    en += p * np.log2(p)

  en = -en
  return en
  

def information_gain(S, S1, S2):

  parent_entropy = entropy(S)
  child1_entropy = entropy(S1)
  child2_entropy = entropy(S2)
  parent_len = len(S)
  child1_len = len(S1)
  child2_len = len(S2)
  left_weight = child1_len / parent_len
  right_weight = child2_len / parent_len

  info_gain = parent_entropy - (child1_entropy * left_weight + child2_entropy * right_weight)

  return info_gain

  

  


def best_split_for_feature(x, y):
  sorted_indices = np.argsort(x)
  x = x[sorted_indices]
  y = y[sorted_indices]
  thresholds = []
  for i in range(len(x) - 1):
    thresholds.append((x[i] + x[i + 1]) / 2)
  best_info_gain = -1
  best_threshold = None
  
  for t in thresholds:
    left_split = y[x < t]
    right_split = y[x >= t]
    if len(left_split) == 0 or len(right_split) == 0:
      continue
    info_gain = information_gain(y, left_split, right_split)
    if info_gain > best_info_gain:
      best_info_gain = info_gain
      best_threshold = t
    elif info_gain == best_info_gain and t < best_threshold:
      best_threshold = t

  return best_threshold, best_info_gain




def best_feature_to_split_on(X, y):
  best_feature_index = None
  best_info_gain = -1
  best_threshold = None
  for feature_index in range(X.shape[1]):
    feature_column = X[:, feature_index]
    threshold, info_gain = best_split_for_feature(feature_column, y)
    if info_gain is not None:
      if info_gain > best_info_gain:
        best_info_gain = info_gain
        best_feature_index = feature_index
        best_threshold = threshold
      elif info_gain == best_info_gain and feature_index < best_feature_index:
        best_feature_index = feature_index
        best_threshold = threshold

  return best_feature_index, best_info_gain, best_threshold


def load_random_dataset():

  """
  loads random dataset

  Returns:
  X (2d np.array): An array of size (n x d) 
  y (1d np.array): The array of labels of size n

  """
  current_file_directory = dirname(abspath(__file__))
  project_root_directory = dirname(current_file_directory)

  X = np.load(join(project_root_directory, 'datasets', 'X_random.npy'))
  y = np.load(join(project_root_directory, 'datasets', 'y_random.npy'))

  return X, y


def load_circle_dataset():

  """
  loads circle dataset


  Returns:
  X_train (2d np.array): An array of size (n x d) 
  y_train (1d np.array): The array of labels of size n
  X_val (2d np.array): An array of size (n x d) 
  y_val (1d np.array): The array of labels of size n

  """

  current_file_directory = dirname(abspath(__file__))
  project_root_directory = dirname(current_file_directory)

  X_train = np.load(join(project_root_directory, 'datasets', 'X_circle_train.npy'))
  y_train = np.load(join(project_root_directory, 'datasets', 'y_circle_train.npy'))
  X_val = np.load(join(project_root_directory, 'datasets', 'X_circle_val.npy'))
  y_val = np.load(join(project_root_directory, 'datasets', 'y_circle_val.npy'))

  return X_train, y_train, X_val, y_val



def plot_decision_boundary(X, y, model, title, resolution=0.02):
   
  """
  Visualizes the decision boundary for a 2D classification problem.

  Parameters:
  - X: Feature matrix (2D, shape: [n_samples, 2])
  - y: Target labels (1D, shape: [n_samples])
  - model: A trained classifier with a predict method
  - resolution: Resolution of the mesh grid (default is 0.02)
  """

  # create figure
  fig, ax = plt.subplots()

  # Define the mesh grid over the feature space
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

  # Predict the class labels for each point in the mesh grid
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  cmap = plt.cm.RdYlBu
  norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(y) + 1.5), ncolors=cmap.N)
  
  # Plot the decision boundary
  ax.contourf(xx, yy, Z, alpha=0.75, cmap=cmap)

  # Plot the points in the dataset
  scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',norm=norm, cmap=cmap, s=50)

  # Add labels and title
  ax.set_xlabel('Feature 0')
  ax.set_ylabel('Feature 1')
  ax.set_title(title)

  # Generate class labels like "Class 0", "Class 1", ...
  unique_classes = np.unique(y)
  class_labels = [f'Class {cls}' for cls in unique_classes]

  # Create custom legend with the class labels
  handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(cls)), markersize=10) for cls in unique_classes]

  # Create the legend
  ax.legend(handles, class_labels, loc='best')

  return fig


def plot_dataset(X, y, title, xlabel='Feature 0', ylabel='Feature 1', legend=True):
   
  """
  Visualizes a dataset

  Parameters:
  - X: Feature matrix (2D, shape: [n_samples, 2])
  - y: Target labels (1D, shape: [n_samples])

  """

  cmap = plt.cm.RdYlBu
  norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(y) + 1.5), ncolors=cmap.N)
  
  # Plot the points in the dataset
  scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',norm=norm, cmap=cmap, s=50)

  # Add labels and title
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)

  # Generate class labels like "Class 0", "Class 1", ...
  unique_classes = np.unique(y)
  class_labels = [f'Class {cls}' for cls in unique_classes]

  # Create custom legend with the class labels
  handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(cls)), markersize=10) for cls in unique_classes]

  # Create the legend
  plt.legend(handles, class_labels, loc='best')

  # Show the plot
  plt.show()


