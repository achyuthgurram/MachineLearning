**Regression with multiple variables**

Multiple features each feature represented as

x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>... x<sub>n</sub>

where n is the number of features.

All training examples will have values for n features.

x<sup>(i)</sup> is the features of i<sup>th</sup> training example.

x<sup>(i)</sup><sub>j</sub> is the value of feature j in i<sup>th</sup> training example.


f<sub>w,b</sub>(X) = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + w<sub>3</sub>x<sub>3</sub> + w<sub>4</sub>x<sub>4</sub> + b

The regression with multiple input variables is called multiple linear regression. This is **not** same as multivariate regression.

**Vectorization** makes it easy to implement multiple linear regression.

When you implment your code using vectorization, it makes your code shorter and also improves the performance. Learning how to write vectorized code will help you take advantage of modern numerical algebra libraries (numpy). GPU is designed for speeding up graphics in computer and vectorized code executes much more quickly in GPU.

![Vectorization](images/vectorization.png)

Numpy library uses parallel hardware in the computer to calculate the dot product very quickly.

NumPy is a library that extends the base capabilities of python to add a richer data set including more numeric types, vectors, matrices, and many matrix functions.

NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). dimension refers to the number of indexes of an array

Data creation routines in NumPy will generally have a first parameter which is the shape of the object. This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result.

Elements of vectors can be accessed via indexing and slicing. NumPy provides a very complete set of indexing and slicing capabilities.

NumPy makes better use of available data parallelism in the underlying hardware. GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. This is critical in Machine Learning where the data sets are often very large.

**Gradient descent for multiple linear regression with vectorization**

![Vectorization](images/gradient_descent_multiple_linear_reg.png)

![Vectorization](images/gradient_descent_calculation.png)

Gradient descent is a great method for minimizing the cost function J to calculate w and b there is one other algorithm only for solving linear regression. It is called **normal equation** method. This solves all in one go without iterations.

This normal equation method doesn't apply to other regression problems. It is also slow when number of features is large (>10000)

Normal equation method may be used in ML libraries that implement linear regression.

 **Terchniques to make gradient descent run faster**

 **Feature scaling:** 

 When you have different input features that take different range of values it will make gradient descent to run slowly. For example in house price prediction lets say we have two different features, one is size of house in sq.ft and second is number of bedrooms. The size of house in sq.ft takes much bigger values compared to number of bedrooms in the house. 
 
 In this case, The weights of size of house takes smaller set of values where as the weights of bedroom takes much bigger range of values. This is because even slightly increasing the weight for the home size, will increase the output by a lot. 
 
 The contour plot for such weights is going to look like an oval shape, where the weights of house size is going to take much smaller range and weights of bedroom is going to take bigger range. When you have a contour plot like this, it will take time for gradient descent to reach to a minimum.

 In this case the features can be rescaled to have values between 0-1, so when we draw a contour plot, the range of weights across all features is going to look pretty even and speeds up gradient descent.

 **How to do scaling**

 **Divide by max:** If x ranges from 300 <= x <= 2000 then divide each value by the maximum possible value 2000, so all values will lie between 0.15 <= x <= 1.

 Second approach to do scaling is **mean normalization**, calculate the mean of all numbers of x. Subtract mean from each of the numbers and divide with (max - min). This will give numbers in between range of -1 and 1.

**z-score normalization:** Calculate mean and standard deviation. For each point calculate (x - mean)/ (std.deviation).

Aim for rescaling in between -1 and 1. But if the numbers are all small it is still okay. However if one feature values range between 0-2 and other feature scales from -100 to 100, which is too large.

![Gradient descent convergence](images/gradient_descent_convergence_rate.png)

In above picture we can see that cost function reaches to certain point after 300 iterations and there after it is flattened. We can also do a convergence test by defining an epsilon, where if the cost function doesn't change much in any iteration then we declare convergence.

Instead of using an epsilon, using the graph is a better approach to determine the convergence.

**Choosing learning rate:** 