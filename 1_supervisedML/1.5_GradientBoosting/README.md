# Intuition for this:

- We calculate initial prediction by first calculating log(odds) for yes/no type binary prediction:
  if 4 are yes and 2 are no then log(odds) =  log(4/2) = 0.7
- Then we use this to calculate the initial prediction:
![Pasted%20image%2020260823095440.png](assets/Pasted%2520image%252020260823095440.png)

- Now we compare it to a threshold like 0.5 and make predictions that all are yes.

![Pasted%20image%2020260823095544.png](assets/Pasted%2520image%252020260823095544.png)

- Now again we calculate the residuals using the predictions and observed value.

## Making Trees:

- Now make a decision tree that predicts the residuals using the feature columns.

> Similarly to [[Gradient Boosting (Regression)]] We usually keep 8 to 32 leaf nodes for predictions in these decision trees.

- if one leaf contains many values then we find the value of the leaf as:
![Pasted%20image%2020260823095801.png](assets/Pasted%2520image%252020260823095801.png)

- Now we predict again using this:
![Pasted%20image%2020260823095839.png](assets/Pasted%2520image%252020260823095839.png)

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

### Example:

![Pasted%20image%2020260822234111.png](assets/Pasted%2520image%252020260822234111.png)


## Future steps:

- Now it again calculates *residuals* using the new prediction model, and does all the above stuff again to get another tree
- All this means that future trees are *learning* from the mistakes of the past trees
  
  ![Pasted%20image%2020260822234440.png](assets/Pasted%2520image%252020260822234440.png)

![Pasted%20image%2020260822234612.png](assets/Pasted%2520image%252020260822234612.png)

- This way we keep adding more trees to reduce the residual (loss) and get closer values to the actual label.

![Pasted%20image%2020260822234658.png](assets/Pasted%2520image%252020260822234658.png)

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

![Pasted%20image%2020260823100736.png](assets/Pasted%2520image%252020260823100736.png)

- This is called the negative log likelyhood

> The better the prediction the larger the log likelyhood, so we multiply it by -1 to make it a loss function which we need to minimize.

- Converting the predicted notation to log(odds) notation for convinience.

![Pasted%20image%2020260823101210.png](assets/Pasted%2520image%252020260823101210.png)

- The derivative of the loss ultimately becomes the residuals:
![Pasted%20image%2020260823101335.png](assets/Pasted%2520image%252020260823101335.png)



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
		![Pasted%20image%2020260823000606.png](assets/Pasted%2520image%252020260823000606.png)

	- Fit a regression decision tree to input as the features and output as the residuals.

	- Now if a leaf has multiple values ending up there, we use this minimization of loss formula to determine the value at the leaf:
		![Pasted%20image%2020260823001022.png](assets/Pasted%2520image%252020260823001022.png)

	- This usually comes out to be average of the values in that leaf, for the loss function as defined above.

	- Now update the F(x) as last F(x) + learning_rate * new decision tree.


# The whole workflow to fit the Gradient boosting Algorithm:

![Pasted%20image%2020260823001410.png](assets/Pasted%2520image%252020260823001410.png)