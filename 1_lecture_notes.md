ML is a subfield of AI

AGI is AI building machines as intelligent as human. Many people believe best way to get to AGI is by using learning algorithms.

AI is creating a lot of value in software industry and many believe it can provide a lot of value in other industries.

Course discusses about different ML algorithms and when each one is appropriate.


ML is field of study, that gives computers the ability to learn without being explicitly programmed.

2 main types of ML algorithms - supervised and unsupervised.

Supervised used in many real world applications.

Best practices to develop a practical, valuable machine learning system.

**Supervised learning**
99% value created by ML today is with supervised learning.

Online advertising is a supervised learning algorithm. We input ads and user information and present it with sample output whether the user will click on the ad or not. In realtime, this algorithm is presented user info and advertisement. The algorithm will predict which ad is more likely to be clicked by the user.

Housing price prediction is a particular type of supervised learning called regression. **Regression** is trying to predict a number from infinitely many possible numbers.

For the housing price prediction based on the size of the home, you can fit a straight line or a curve or even a more complex model.

Second type of supervised learning is **classification**. One example is breast cancer detection.
Build a machine learning algorithm so doctors can have a diagnosis tool to detect breast cancer.

Using ML algorithms, to detect if a tumor is cancerous or dangerous or if the tumor is benign.

Dataset should have various features and a label if the tumor is malignant (1) or benign (0).

In classification we have set of possible outputs. Different from regression where the output can have infinite possible numbers.

In classification, there can also be more than two possible outputs. Example in breast cancer detection you can have type 1 and type 2 cancer depending on size of tumor and also benign tumor.

classification predict categories. classification predicts small, finite limited set of possible output categories like 0, 1, 2 but not like 0.5, 1.6 etc...

supervised learning maps from input X to output Y where the learning algorithm learns from the right answers.

**Unsupervised learning**

After supervised learning, most widely used ML algorithm is unsupervised learning.

In unsupervised learning, we are not given any labeling of data in the input sample. Instead we are asked to identify patterns based on the given data. In supervised learning we label the data to train the algorithm but in unsupervised the input is unlabeled data.

The algorithms for unsupervised clusters the data. It assigns the data into different clusters.

Clustering algorithm is used in Google news every day. It groups related stories together. It runs on every day news articles and groups them together. Algorithm has to figure out on it's own without supervision.

DNA microarray data, spreadsheet where each column represents DNA activity of one person. Each row represents a particular gene. One gene affects eye color, another gene effects how tall they are, Another gene effects like or dislike to something.

Run clustering algorithm to group people together based on their genes. This is also unsupervised because we are just grouping the data without any classification.

Companies have huge database of customer information. Group customers together so you can better serve the customers.

Clustering algorithm which is unsupervised learning algorith, takes data without labels and tries to automatically group them into clusters.

In supervised learning, data comes with input X and output label Y. In unsupervised learning data comes wtih input X but no output lavel Y. Algorithm has to find structure in the data.

**Clustering** groups similar data together.

**Anomaly detection** another unsupervised learning, finds unusual data points. useful in fraud reduction in financial system.

**Dimensionality reduction** another unsupervised learning, takes a big dataset and magically compresses it to smaller dataset while loosing as little information as possible.

Jupyter notebooks have markdown cells which is a bunch of text. Something like a description of what you are trying to do.

Code cell creates a new cell where you can enter code and execute it.

Linear regression - fitting a straight line to your data.

Any supervised model that predicts numbers is a regression problem.

There are other models for addressing regression problem.

The dataset used to train the model is called training set.

For the housing example where you predict the house price based on the size of the house, the training data set has **features** also called as x which is the square foot of the house, price also called as y is the **output** or **target** variable. In the training data set each row contains a different training example.

To denote a single training example use the notation (x, y).

Linear regression with one variable - size of home - univariate linear regression.

![Linear regression](images/linear_regression.png)

NumPy, a popular library for scientific computing

Matplotlib, a popular library for plotting data

Below line creates numpy arrays. When using f-string output formatting, the content between the curly braces is evaluated when producing the output.

```
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```

Numpy arrays have a .shape parameter. x_train.shape returns a python tuple with an entry for each dimension

Plot these two points using the scatter() function in the matplotlib library

```
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```

**Cost function**

Cost function will tell us how well the model is doing, so we can improve the model.

f(x) = wx + b

w, b are called parameters or coefficients or weights.

Squared error cost function is the most common cost function for linear regression problem.

The result of squared error is divided by m so that the result of the function doesn't increase with the number of training examples. It is also a common practice to divide by 2m instead of m but it not really necessary.

The purpose of the cost function is to find values for parameter w and b that minimize the error

![Squared error cost function](images/squared_error_cost_function.png)


![Cost function goal](images/cost_function_goal.png)

See below cost function analysis where we only have parameter w and not b and only the analysis is done on parameter w. On the left is the linear regression model and on the right is the cost function. At w = 1 the cost function has minimum error. This helps us choose a good value for w.

![Cost function analysis](images/cost_function_analysis.png)

Sample cost function analysis below with w and b values with a 3D plot.

![Cost function analysis](images/cost_function_analysis_with_wb.png)

Another way of visualizing cost function is using a contour plot. It shows all the points that are at same height for different heights. All points in the same height are connected as a circle or line.

The contour plot is represented on a 2d space instead of the 3d space. The contour plot on the top right, each of the ovals indicate the set of points on 3d space which are at the exact same height i.e. the same value for cost function.

Contour plots are a convenient way to visualize the 3D cost function by plotting it in 2D.

![Contour plot](images/contour_plot.png)

In real world the models are much more complex and it is not easy to find the values of w and b that minimizes the cost function using contour plot. Instead there is an algorithm called gradient descent which can be used for this purpose.

Gradient descent and variations on gradient descent are used to train not just the linear regression, but some of the biggest and most complex models in all of AI.

**Gradient descent**

This is used for all over the place in ML. Not just for linear regression but also for training neural networks like deep learning models.

It is used to minimize cost function for any function not just for linear regression.

It can be used for minimizing cost function with multiple parameters like 
J(w1, w2, w3, w4, ... , b)

It is possible that a complex J may not have one minimum, but have multiple minimum.

![Gradient descent](images/gradient_descent.png)

Gradient descent takes you to the local minimum which may not be the maximum minimum possible. It depends on where you start and which direction you take to reach the minimum.

In gradient descent there is a parameter called learning rate which controls how big of a step we take downhill.

If learning rate is small we take baby steps, if learning rate is high we take quick steps.

Repeat below steps until convergence i.e. we reach local minimum and w and b won't change much with each additional step we take

When someone talks about gradient descent they always mean simultaneous update of parameters.

Use the correct version of simultaneous update and not the incorrect way.

![Gradient descent](images/gradient_descent_convergence.png)

Below shows how the derivative term helps gradient descent change w to get you closer to the minimum

![Gradient descent](images/gradient_descent_intution.png)

The choice of learning rate will have a huge impact on the implementation of gradient descent algorithm. If learning rate is choosen poorly, then gradient descent may not work at all.

If learning rate alpha is too small, gradient descent will work but it will take a lot of time and it will be very slow.

If learning rate alpha is too large, then the gradient descent will take huge steps going further and further away from the minimum. Gradient descent may fail to converge in this case.

When you reach the local min, further gradient descent steps will not update weight. The tangent or slope at the local minimum is 0, the derivative becomes zero, so the weight is unchanged.

![Gradient descent](images/gradient_descent_local_min.png)

Gradient descent can reach local minimum with a fixed learning rate.

As we approach the minimum the derivative becomes smaller and smaller i.e. the slope will become smaller. i.e the update steps become smaller when it is reaching the local minimum.

![Gradient descent](images/gradient_descent_smaller_derivative.png)

Gradient descent can be used to minimize any cost function J not just the mean squared error cost function we use for linear regression.

![Gradient descent](images/linear_regression_cost_function_gradient.png)


![Partial derivative](images/partial_derivative.png)

squared error cost function used for linear regression has only one minimum which is global minimum and no other local minimum. Gradient descent will always take you to the global minimum.

![Partial derivative](images/gradient_descent_regression.png)

"Batch" gradient descent - each step of gradient descent uses all the training examples. There are other gradient descent can only use a subset of training examples.



**TODO**

Practice quiz for machine learning lectures

