## Binary Logistic Regression Classifier

### Overview


The program LogRegNum will

* load data from the file "rawData1.txt"
* randomize the examples of the data
* split the data into a training set and a cross validation set
* give basic descriptive statistics of the training set predictors
* produce a scatter plot, histograms, and box plots for the training set 
* build a predictive model (via Logistic Regression) from the training set 
* test the prediction on the cross validation set
* produce scatter plots of the training set, cross validation set, and the decision boundary


### Ipython (Jupyter) Notebook

This is contained in the file **LogRegNum.ipynb** and the exported pdf is **LogRegNum.pdf**

### Interactive Version

To run the program type **python LogRegNum.py** at the terminal prompt. The program will ask "Are you in a hurry? [y/n]:" 

A response of **y** will result in the program producing output immediately with minimal pauses. A response of **n** will result reveal the output incrementally with longer pauses; this is recommended when running the program for the first time. 

An example of the terminal output is given below.

> 
Welcome to our Binary Linear Regression Classifier Demo!
>
Are you in a hurry? [y/n]: y
>
Ok, there are three parts to the output. 
>
Press ENTER when you are ready for Part 1.
>
The data has been loaded and randomized.
The data has 100 examples and 3 features. 
>
The features are
'Evaluation 1', 'Evaluation 2', 'Disease State'. 
>
The first 10 examples of the data
[[ 60.458  73.095   1.   ]
 [ 80.19   44.822   1.   ]
 [ 50.286  49.805   0.   ]
 [ 42.262  87.104   1.   ]
 [ 47.264  88.476   1.   ]
 [ 55.34   64.932   1.   ]
 [ 95.862  38.225   0.   ]
 [ 72.346  96.228   1.   ]
 [ 76.979  47.576   1.   ]
 [ 53.971  89.207   1.   ]]
>
Training set: 60 examples (60% of the data).
Cross Validation set: 40 examples (the other 40%). 
>
There are 37 positive training examples and 23 negative training examples.
We don't have skewed classes. 
>
Let's look at a scatterplot of the training examples
Please close the plot to continue.
Thanks! 
>
Press ENTER when you are ready for Part 2.
>
Statistics of 'Evaluation 1': (Numbers are rounded for easy reading)
The mean is 66.
The standard deviation is 19.
The range is 32 to 99. 
>
Statistics of 'Evaluation 2': (Numbers are rounded for easy reading)
The mean is 67.
The standard deviation is 18.
The range is 38 to 97. 
>
Let's plot histograms and boxplots of 'Evaluation 1' and 'Evaluation 2'
Please close the plots to continue.
Thanks! 
>
Press ENTER when you are ready for Part 3.
>
A logistic regression model will be used.
The regularization parameter is C = 100. 
>
How good is our model on the training set?
The training accuracy is 91%.
Good. 
>
How about the cross validation set?
The cross validation accuracy is 87%.
Not bad. 
>
Finally, let's plot the training and cross validation sets.
Please close the plots when you're done.
Thanks! 

Plots appear in separate windows. The plots pause the program, so the plots must be closed in order to resume. These plot-pauses define the boundaries for parts 1, 2 and 3.










