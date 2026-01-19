In classification, the output only takes a handful of possible values instead of any number.

Logistic regression algorithm used for classification.

Example: Email is spam or not, Transaction fraudulent, Tumor malignant. In each of these problems, the result can be yes or no.

The answers can also be called 0 or 1, false or true, negative or positive.

In email spam example, negative vs positive means absence of spam in it vs positive means presence of spam in it.

![Linear regression for classification](images/linear_regression_for_classification.png)

Picture above shows using linear regression for classification problem. The initial blue line which came out of linear regression is okay fit to the classification problem, considering all left of the blue line as benign and all right as malignant. 

However if we consider one more data point, the linear regression model shifts the decision boundary to the right, see green line. This is not a good model because it misclassified many tumor sizes which are malignant as benign.

Logistic regression is used for binary classification. It is the widely used classification algorithm.

Logistic regression will fit a s shaped curve to the dataset. Logistic regression will tell how closer the tumor has a chance of being malignant.

sigmoid function also called logistic function. The horizontal axis takes on both positive and negative values. The output of sigmoid function between 0 and 1.

![Sigmoid function](images/sigmoid_function.png)

Using sigmoid function to build logistic regression. Sigmoid function maps all input values to values between 0 and 1.

![Sigmoid for logistic regression](images/sigmoid_for_logistic_regression.png)

![Logistic regression for classification](images/logistic_regression.png)
