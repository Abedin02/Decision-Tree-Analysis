This project explores how varying the maximum depth of a Decision Tree classifier affects:
-Decision boundaries (visual intuition of model complexity)
-Model performance (underfitting → optimal → overfitting)
The goal is to understand how depth controls the trade-off between model complexity and generalization.
Decision Trees are highly flexible models that can fit complex decision functions, but this flexibility can also cause overfitting.
To study this, we evaluate decision trees with different max_depth settings:
-Unconstrained tree (max_depth = None) -> overfitting
-Shallow trees (max_depth = 1, simple rules) -> underfitting
-Medium-depth trees (balanced complexity)
-Deep trees (complex boundaries)
-Wide hyperparameter sweep (max_depth = 0–100)
We compute scores (accuracy) across depth values from 0 to 100 (hyperparameter tuning curve):
-Depth too low → underfitting (low bias, high error)
-Optimal depth → best generalization
-Depth too high → overfitting (high variance)
The tuning curve shows where performance peaks.

For each model, we plot how the classification boundary changes as max_depth increases, 
which helps visualize model complexity as depth increases.
