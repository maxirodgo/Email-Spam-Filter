# Email-Spam-Filter
Email Spam Classifier on Python 
	
	
	import os
	from pylab import *
	import matplotlib.pyplot as plt
	%matplotlib inline 
	
	from helper import *



	np.random.seed(12)
	n_samples = 500

	class_one = np.random.multivariate_normal([5, 10], [[1, .25],[.25, 1]], n_samples)
	class_one_labels = -np.ones(n_samples)

	class_two = np.random.multivariate_normal([0, 5], [[1, .25],[.25, 1]], n_samples)
	class_two_labels = np.ones(n_samples)

	features = np.vstack((class_one, class_two))
	labels = np.hstack((class_one_labels, class_two_labels))

# What our feature arrays look like
	features

# Visualize these data distributions
	plt.figure(figsize=(9, 6))
	plt.scatter(features[:, 0], features[:, 1],
            c = labels, alpha = .6);

	plt.title("Binary labeled data in 2D", size=15);
	plt.xlabel("Feature 1", size=13);
	plt.ylabel("Feature 2", size=13);

# Sigmoid Function 

	def sigmoid(z):
    	 # Input: 
   	 # z : scalar or array of dimension n 
   	 # Output:
   	 # sgmd: scalar or array of dimension n
    
   	 sgmd = 1 / (1 + np.exp(-z))
    
    
    	 return sgmd
    
    
# Implement binary Logistic Regression 

	def y_pred(X, w, b=0):
    	# Input:
    	# X: nxd matrix
    	# w: d-dimensional vector
    	# b: scalar (optional, if not passed on is treated as 0)
    	# Output:
    	# prob: n-dimensional vector
    
    	probability = sigmoid((X@w + b))
    
    	return probability
    
    
# Computing Negative Loss Likelihood   
    
	def log_loss(X, y, w, b=0):
   	 # Input:
   	 # X: nxd matrix
   	 # y: n-dimensional vector with labels (+1 or -1)
   	 # w: d-dimensional vector
   	 # Output:
   	 # a scalar
    	 assert np.sum(np.abs(y))==len(y) # check if all labels in y are either +1 or -1
    
   	 n11= np.sum(np.log(1+np.exp(-y*(X@w + b))))
    
    	 return n11
    
# Computing Gradient 
    
	def gradient(X, y, w, b):
         # Input:
         # X: nxd matrix
         # y: n-dimensional vector with labels (+1 or -1)
   	 # w: d-dimensional vector
   	 # b: a scalar bias term
  	  # Output:
   	 # wgrad: d-dimensional vector with gradient
   	 # bgrad: a scalar with gradient
    
   	 n, d = X.shape
    	 wgrad = np.zeros(d)
    	 bgrad = 0.0
    	 grad = (-y*sigmoid(-y*(X@w + b)))
   	 wgrad= X.T@grad
    	 bgrad= np.sum(grad)
    	 return wgrad, bgrad
    
# Weight Update of Gradient Descent
    
   	def logistic_regression(X, y, max_iter, alpha):
    	n, d = X.shape
    	w = np.zeros(d)
    	b = 0.0
    	losses = np.zeros(max_iter)    
    
    for step in range(max_iter):
        wgrad, bgrad = gradient(X, y, w, b)
        w -= alpha * wgrad
        b -= alpha * bgrad
        losses[step] = log_loss(X, y, w, b)
    return w, b, losses\ 

	weight, b, losses = logistic_regression(features, labels, 1000, 1e-04)
	plot(losses)
	xlabel('iterations')
	ylabel('log_loss') 
    
# Running Implementation of Data To Check Loss Against Iterations 

	max_iter = 10000
	alpha = 1e-4
	final_w, final_b, losses = logistic_regression(features, labels, max_iter, alpha)

	plt.figure(figsize=(9, 6))
	plt.plot(losses)
	plt.title("Loss vs. iteration", size=15)
	plt.xlabel("Num iteration", size=13)
	plt.ylabel("Loss value", size=13)

# Plotting the decision boundary 

	scores = y_pred(features, final_w, final_b)

	pred_labels = (scores > 0.5).astype(int)
	pred_labels[pred_labels != 1] = -1

	plt.figure(figsize=(9, 6))

	x = np.linspace(np.amin(features[:, 0]), np.amax(features[:, 0]), 10)
	y = -(final_w[0] * x + final_b)/ final_w[1] 
	plt.plot(x, y)

	plt.scatter(features[:, 0], features[:, 1],

	train_url = 's3://codio/CIS530/CIS533/data_train'
	test_url = 's3://codio/CIS530/CIS533/data_test'= pred_labels, alpha = .6)
	plt.title("Predicted labels", size=15)
	plt.xlabel("Feature 1", size=13)
	plt.ylabel("Feature 2", size=13)
	plt.axis([-3,10,0,15])

# Building Email Classifier

	import pandas as pd
	import dask
	import dask.bag
	from dask.diagnostics import ProgressBar


	train_url = 's3://codio/CIS530/CIS533/data_train'
	test_url = 's3://codio/CIS530/CIS533/data_test'

# Tokenize the email and hashes the symbols into a vector
	def extract_features_naive(email, B):
	    # initialize all-zeros feature vector
	    v = np.zeros(B)
	    email = ' '.join(email)
	    # breaks for non-ascii characters
	    tokens = email.split()
	    for token in tokens:
        v[hash(token) % B] = 1
    return v

	def load_spam_data(extract_features, B=512, url=train_url):
	    '''
	    INPUT:
	    extractfeatures : function to extract features
	    B               : dimensionality of feature space
	    path            : the path of folder to be processed

	    OUTPUT:
	    X, Y
	    '''

	    all_emails = pd.read_csv(url+'/index', header=None).values.flatten()

	    xs = np.zeros((len(all_emails), B))
	    ys = np.zeros(len(all_emails))

	    labels = [k.split()[0] for k in all_emails]
	    paths = [url+'/'+k.split()[1] for k in all_emails]

	    ProgressBar().register()
	    dask.config.set(scheduler='threads', num_workers=50)
	    bag = dask.bag.read_text(paths, storage_options={'anon':True})
	    contents = dask.bag.compute(*bag.to_delayed())
	    for i, email in enumerate(contents):
		# make labels +1 for "spam" and -1 for "ham"
		ys[i] = (labels[i] == 'spam') * 2 - 1
		xs[i, :] = extract_features(email, B)
	    print('Loaded %d input emails.' % len(ys))
	return xs, ys

	Xspam, Yspam = load_spam_data(extract_features_naive)
	Xspam.shape

# Split the Data 

	#Split data into training (xTr and yTr) and testing (xTv and yTv)
	n, d = Xspam.shape
	#Allocate 80% of the data for training and 20% for testing
	cutoff = int(np.ceil(0.8 * n))
	#indices of training samples
	xTr = Xspam[:cutoff,:]
	yTr = Yspam[:cutoff]
	#indices of testing samples
	xTv = Xspam[cutoff:]
	yTv = Yspam[cutoff:]

# Training and Evalutaion 

	max_iter = 5000
	alpha = 1e-5
	final_w_spam, final_b_spam, losses = logistic_regression(xTr, yTr, max_iter, alpha)

	plt.figure(figsize=(9, 6))
	plt.plot(losses)
	plt.title("Loss vs. iteration", size=15)
	plt.xlabel("Num iteration", size=13)
	plt.ylabel("Loss value", size=13)

	#evaluate training accuracy
	scoresTr = y_pred(xTr, final_w_spam, final_b_spam)
	pred_labels = (scoresTr > 0.5).astype(int)
	pred_labels[pred_labels != 1] = -1
	trainingacc = np.mean(pred_labels == yTr)

	#evaluate testing accuracy
	scoresTv = y_pred(xTv, final_w_spam, final_b_spam)
	pred_labels = (scoresTv > 0.5).astype(int)
	pred_labels[pred_labels != 1] = -1
	validationacc = np.mean(pred_labels==yTv)
	print("Training accuracy %2.2f%%\nValidation accuracy %2.2f%%\n" % (trainingacc*100,validationacc*100))

    
