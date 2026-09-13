# Intuition for this:

- We calculate initial prediction by first calculating log(odds) for yes/no type binary prediction:
  if 4 are yes and 2 are no then log(odds) =  log(4/2) = 0.7

- Now we compare it to a threshold like 0.5 and make predictions that all are yes.


- Now again we calculate the residuals using the predictions and observed value.

## Making Trees:

- Now make a decision tree that predicts the residuals using the feature columns.

> Similarly to [[Gradient Boosting (Regression)]] We usually keep 8 to 32 leaf nodes for predictions in these decision trees.

- if one leaf contains many values then we find the value of the leaf as:

- Now we predict again using this:

- This will only give a new log(odds) value
  we have to then use the probability formula to calculate the probablity again.
  
Now calculate residuals again and repeat the whole process to make new trees.

---

## Basic Intuition (Understanding for regression):

- it _starts_ with a _single leaf_ which only _predicts the average value_ of the output values in training set.
- It then calculates ==residuals== = observed value - predicted value.
- It then Expands the tree to _predict_ the ==residuals== using all the feature data.
- The leaf values are then averaged according to the values present at each leaf.

> **Note:**
> Usually the total number of leaves in the tree is only allowed (set) between 8 and 32 in real world applications.

- Now to predict values, it uses:    105\
  ```python
  prediction = average_output + learning_rate * (Decision tree output)
  ```

> Learning rate is between (0, 1)
> *This is used to achieve lower variance.*



## Future steps:

- Now it again calculates *residuals* using the new prediction model, and does all the above stuff again to get another tree
- All this means that future trees are *learning* from the mistakes of the past trees


- This way we keep adding more trees to reduce the residual (loss) and get closer values to the actual label.


> Full mathematical model is in [[Maths Behind Gradient Boosting Regression]]

---

## Dataset Used for Implementation:

- Kaggle August Playground Dataset: For smartphone addiction prediction (A classification problem)
- [Kaggle Dataset](https://www.kaggle.com/competitions/playground-series-s6e8/overview)

## Learnings During application:

- The feature values need not be standardised/normalized because xgboost just creates multiple decision trees and decision trees don't need standardised values.

## Results:

- Final R2 Score: 0.658
- average percentage of error in prediction: ~ 20%

> From community discussions i saw that the most we can pull is till 0.67 r2 score using optimized models.
> Even i used many ==hyperparameter tuning== in the xgboost to raise my r2 score

*finally*:
I think that the score is pretty good for a default xgboost.

---

- We get intuition behind [[Gradient Boosting (Classification)]], but now here's the Mathematical WHY?

# Maths:

*few Terms*:
- F(x) is the prediction model
- L(y, F(x)) is the loss function 

> Here loss function is different from regression because classification has binary labels.

- This is called the negative log likelyhood

> The better the prediction the larger the log likelyhood, so we multiply it by -1 to make it a loss function which we need to minimize.

- Converting the predicted notation to log(odds) notation for convinience.

- The derivative of the loss ultimately becomes the residuals:

# Steps to make the decision trees:

- All the steps are now similar to [[Maths Begind Gradient Boosting Classificaiton]] after defining loss function and residuals for this classification problem.

---

From [[Gradient Boosting (Regression)]], I got the intuition behind the algorithm.

- Now we get the mathemtical "WHY?":

## Maths behind Gradient boosting:

*Few terms:*
1. Prediction Function => F(x) = gives prediction
2. Loss function:
	- L(yi, F(xi)) = 1/2(observed - predicted)^2

### Now Steps:

1.  Initialization:
	- Initial Prediction: F(x) = average of all labels
	- this is because taking loss over all data points and minimizing the loss gets us F(x) as average of all labels

2. make a loop for m = 1 to M:
	- compute residuals:

	- Fit a regression decision tree to input as the features and output as the residuals.

	- Now if a leaf has multiple values ending up there, we use this minimization of loss formula to determine the value at the leaf:

	- This usually comes out to be average of the values in that leaf, for the loss function as defined above.

	- Now update the F(x) as last F(x) + learning_rate * new decision tree.
