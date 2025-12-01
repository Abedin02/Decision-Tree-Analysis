import numpy as np


class DecisionTreeNode:

  """
  object to represent a decision tree node

  """

  def __init__(
    self, 
    is_leaf, 
    split_idx=None, 
    split_val=None, 
    entropy=None,
    label=None, 
    left_node=None,
    right_node=None
  ):

    # boolean to indicate if node is a leaf node
    self.is_leaf = is_leaf

    # column index of feature that node splits on
    self.split_idx = split_idx
    self.entropy = entropy

    # output label (None if not a leaf node)
    self.label = label

    # value of feature the node splits on (splits to left if <= to value)
    self.split_val = split_val

    # pointer to left and right node
    self.left_node = left_node
    self.right_node = right_node


  def __repr__(self):

    """
    string representation of decision tree node

    """

    if self.is_leaf:
      return f'Leaf Node - Label {self.label}'
    else:
      return f'Split Node: Feature {self.split_idx} < {np.round(self.split_val,2)}\nEntropy={np.round(self.entropy,3)}'

