# # Binary Classifier using Linear Regression

# load python modules

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import linear_model

print ""
print "Welcome to our Binary Linear Regression Classifier Demo!"
time.sleep(2)

print ""
hurry = raw_input("Are you in a hurry? [y/n]: ")
time.sleep(1)

if  hurry == 'y':
    print ""
    print "Ok, there are three parts to the output. \n "
    t = np.zeros(10)
elif hurry == 'n':
    print ""
    print "Ok, let me walk you through the analysis. \n"
    t = np.arange(10)
else:
    print ""
    print "I didn't get that. Please try again."
    print ""
    quit()
        
raw_input("Press ENTER when you are ready for Part 1.")
time.sleep(3)
    
# load the data and randomize the examples

data = np.loadtxt('rawData1.txt', delimiter=',')
np.random.shuffle(data) 
print ""
print "The data has been loaded and randomized."
time.sleep(t[2])


# dimensions of the data

[k,l] = data.shape 
print "The data has %i examples and %i features. \n" %(k,l)
time.sleep(t[3])

print "The features are"
time.sleep(t[1])
print "'Evaluation 1', 'Evaluation 2', 'Disease State'. \n"
time.sleep(t[3])


# take a peek at the data

print "The first 10 examples of the data"
print data[:10,:]
print ""


if hurry == 'y':
    pass
else:
    raw_input("Press ENTER to continue \n")
    
    
time.sleep(t[2])


# determine number of training examples and cross validation examples

m = int(np.ceil(0.60 * k)) # m=(number of training examples)
print "Training set: %i examples (60%% of the data)." %m
time.sleep(t[3])
print "Cross Validation set: %i examples (the other 40%%). \n" %(k-m)
time.sleep(t[3])


# define training set

dataTrain = data[:m,:] 


# define cross validation set

dataCV = data[m:,:] 


# define X (training predictors)

X = dataTrain[:, :2] 


# define y (training responses)

y = dataTrain[:, 2] 


# define Xcv (cross validation predictors)

Xcv = dataCV[:, :2] 


# define ycv (cross validation responses)

ycv = dataCV[:, 2] 


# split up positive/negative training examples

pos = dataTrain[y == 1]; neg = dataTrain[y == 0] 


# split up positive/negative cross validation examples

poscv = dataCV[ycv == 1]; negcv = dataCV[ycv == 0] 


# check for skewed classes 

[P, N] = [pos.shape[0] , neg.shape[0]]
print "There are %i positive training examples and %i negative training examples." %(P,N)
time.sleep(t[3])

if max(float(P)/m, float(N)/m) >= 80:
    print "We have skewed classes. \n"
else:
    print "We don't have skewed classes. \n"
time.sleep(t[3])


# look at a scatterplot of the training examples

print "Let's look at a scatterplot of the training examples"
time.sleep(t[2])

print "Please close the plot to continue."
time.sleep(t[1])

print "Thanks! \n"
time.sleep(t[2])

# Plot the training points
plt.figure()
pscatt = plt.scatter(pos[:,0], pos[:,1], color = 'c', marker = 'o', edgecolors='k')
nscatt = plt.scatter(neg[:,0], neg[:,1], color = 'm', marker = 'o', edgecolors='k')
plt.xlabel('Evaluation 1')
plt.ylabel('Evaluation 2')
plt.legend([pscatt, nscatt], ['Diseased', 'Not Diseased'], loc="lower center", ncol=2)
plt.title('Scatter Plot of Training Examples')
x_min, x_max = X[:,0].min() - 10, X[:,0].max() + 10
y_min, y_max = X[:,1].min() - 25, X[:,1].max() + 20
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()
plt.close()

time.sleep(2)

raw_input("Press ENTER when you are ready for Part 2.")
print ""
time.sleep(1)

# statistics of 'Evaluation 1'

print "Statistics of 'Evaluation 1': (Numbers are rounded for easy reading)"
time.sleep(t[2])

mu1 = np.mean(X[:,0])
print "The mean is %i." %mu1
time.sleep(t[2])

s1 = np.std(X[:,0])
print "The standard deviation is %i." %s1
time.sleep(t[2])

[min1, max1] = [np.min(X[:,0]), np.max(X[:,0])]
print "The range is %i to %i. \n" %(min1,max1)
time.sleep(t[3])


# statistics of 'Evaluation 2'

print "Statistics of 'Evaluation 2': (Numbers are rounded for easy reading)"
time.sleep(t[2])

mu2 = np.mean(X[:,1])
print "The mean is %i." %mu2
time.sleep(t[2])

s2 = np.std(X[:,1])
print "The standard deviation is %i." %s2
time.sleep(t[2])

[min2, max2] = [np.min(X[:,1]), np.max(X[:,1])]
print "The range is %i to %i. \n" %(min2,max2)
time.sleep(t[3])


# Histograms/Boxplots of 'Evaluation 1' and 'Evaluation 2'

print "Let's plot histograms and boxplots of 'Evaluation 1' and 'Evaluation 2'"
time.sleep(t[2])

print "Please close the plots to continue."
time.sleep(t[1])

print "Thanks! \n"
time.sleep(t[2])


# Histogram for 'Evaluation 1'
plt.figure(1)

# create parameters for histogram
n, bins, patches = plt.hist(X[:,0], 20, normed=1, facecolor = 'g', cumulative=False)

# plot options
plt.xlabel('Evaluation 1')
plt.ylabel('Relative Frequency')
plt.title("Histogram of 'Evaluation 1'")
plt.axis([30, 100, 0, 0.05])
plt.grid(True)

# add a normal curve
norms = mlab.normpdf(bins, mu1, s1)
plt.plot(bins, norms, 'k--', linewidth=1)


# Histogram for 'Evaluation 2'
plt.figure(2)

# create parameters for histogram
n, bins, patches = plt.hist(X[:,1], 20, normed=1, facecolor = 'b', cumulative=False)

# plot options
plt.xlabel('Evaluation 2')
plt.ylabel('Relative Frequency')
plt.title("Histogram of 'Evaluation 2'")
plt.axis([30, 100, 0, 0.05])
plt.grid(True)

# add a normal curve
norms = mlab.normpdf(bins, mu2, s2)
plt.plot(bins, norms, 'k--', linewidth=1)

# box plots of 'Evaluation 1' and 'Evaluation 2'
plt.figure(3)

bp = plt.boxplot(X, notch=True, vert=False, showmeans=True, labels=['Evaluation 1', 'Evaluation 2'], patch_artist=True)
colors=['green', 'blue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.title("Boxplots of 'Evaluation 1' and 'Evaluation 2'")

plt.show()
plt.close()
time.sleep(1)


raw_input("Press ENTER when you are ready for Part 3.")
print ""
time.sleep(1)


# set the regularization parameter (C value)

regpar = 1.0e2 


# logistic regression model with specified C

logreg = linear_model.LogisticRegression(C=regpar) 
print "A logistic regression model will be used."
time.sleep(t[2])
print "The regularization parameter is C = %i. \n" %regpar
time.sleep(t[3])


# fit the logistic regression model

logreg.fit(X,y) 


# training accuracy

print "How good is our model on the training set?"
time.sleep(t[3])

ta = logreg.score(X,y)
ta = ta * 100 # convert to percentage
 
print "The training accuracy is %i%%." %ta
time.sleep(t[3])


# comment on training accuracy

if ta >= 90:
    comment1 = "Good."
elif ta >= 85:
    comment1 = "Not bad."
elif ta >= 75:
    comment1 = "A bit low."
else:
    comment1 = "Very low."
print "%s \n" %comment1    
time.sleep(t[3])


# cross validation accuracy

print "How about the cross validation set?"
time.sleep(t[3])

cva = logreg.score(Xcv,ycv)
cva = cva * 100 # convert to percentage

print "The cross validation accuracy is %i%%." %cva
time.sleep(t[2])


# comment on cross validation accuracy

if cva >= 90:
    comment2 =  "Good."
elif cva >= 85:
    comment2 = "Not bad."
elif cva >= 75:
    comment2 = "A bit low."
else:
    comment2 = "Very low."
print "%s \n" %comment2    
time.sleep(t[3])


# get ready for the plots

print "Finally, let's plot the training and cross validation sets."
time.sleep(t[3])

print "Please close the plots when you're done."
time.sleep(t[2])

print "Thanks! \n"
time.sleep(t[1])


# Plots for Training Examples and Cross Validation Examples

# step size in the mesh plot
h = .15 

# plot the decision boundary by assigning a color to each point in the mesh plot
x_min, x_max = X[:,0].min() - 10, X[:,0].max() + 10
y_min, y_max = X[:,1].min() - 25, X[:,1].max() + 20
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the training examples
plt.figure(1)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
pscatt = plt.scatter(pos[:,0], pos[:,1], color = 'c', marker = 'o', edgecolors='k')
nscatt = plt.scatter(neg[:,0], neg[:,1], color = 'm', marker = 'o', edgecolors='k')
plt.xlabel('Evaluation 1')
plt.ylabel('Evaluation 2')
plt.legend([pscatt, nscatt], ['Diseased', 'Not Diseased'], loc="lower left", ncol=2)
plt.title('Training Examples with Decision Boundary \n Prediction Accuracy: %i%% (%s)' %(ta, comment1))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Plot the cross validation examples
plt.figure(2)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
pscatt = plt.scatter(poscv[:,0], poscv[:,1], color = 'c', marker = 's', edgecolors='k')
nscatt = plt.scatter(negcv[:,0], negcv[:,1], color = 'm', marker = 's', edgecolors='k')
plt.xlabel('Evaluation 1')
plt.ylabel('Evaluation 2')
plt.legend([pscatt, nscatt], ['Diseased', 'Not Diseased'], loc="lower left", ncol=2)
plt.title('Cross Validation Examples with Decision Boundary \n Prediction Accuracy: %i%% (%s)' %(cva, comment2))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Show the plots
plt.show()