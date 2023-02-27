import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegularizedLogisticRegression(object):
    '''
    Implement regularized logistic regression for binary classification.

    The weight vector w should be learned by minimizing the regularized loss
    \l(h, (x,y)) = log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
    function that we are trying to minimize is the log loss for binary logistic regression 
    plus Tikhonov regularization with a coefficient of \lambda.
    '''
    def __init__(self):
        self.learningRate = 0.00001 # Feel free to play around with this if you'd like, though this value will do
        self.num_epochs = 100 # Feel free to play around with this if you'd like, though this value will do
        self.batch_size = 15 # Feel free to play around with this if you'd like, though this value will do
        self.weights = None 

        #####################################################################
        #                                                                    #
        #    MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUBMITTING    #
        #                                                                    #
        #####################################################################

        self.lmbda = 100 # tune this parameter

    def train(self, X, Y):
        '''
        Train the model, using batch stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''
        self.weights = np.zeros(X.shape[1])
        for n in range(self.num_epochs):
            print("training...", n)
            p = np.random.permutation(len(X))
            X = X[p]
            Y = Y[p]
            for i in range((int)(len(Y)/self.batch_size-1)):
                X_batch = X[i*self.batch_size: (i+1)*self.batch_size]
                Y_batch = Y[i*self.batch_size: (i+1)*self.batch_size]
                d_loss = np.zeros(self.weights.shape)
                for j in range(X.shape[1]):
                    for x, y in zip(X_batch, Y_batch):
                        d_loss[j] += (sigmoid_function(np.dot(self.weights, x))-y)*x[j]
                    d_loss[j] = d_loss[j] / self.batch_size
                    d_loss[j] += 2 * self.lmbda * self.weights[j]
                self.weights = self.weights - (self.learningRate*d_loss)/len(X_batch)

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        preds = np.asarray([1 if a > 0.5 else 0 for a in sigmoid_function(np.dot(self.weights, np.transpose(X)))])
        return preds

    def accuracy(self,X, Y):
        '''
        Output the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        ypred = self.predict(X)
        return np.sum(ypred == Y)/len(Y)

    def runTrainTestValSplit(self, lambda_list, X_train, Y_train, X_val, Y_val):
        '''
        Given the training and validation data, fit the model with training data and test it with
        respect to each lambda. Record the training error and validation error, which are equivalent 
        to (1 - accuracy).

        @params:
            lambda_list: a list of lambdas
            X_train: a 2D Numpy array for trainig where each row contains an example,
            padded by 1 column for the bias
            Y_train: a 1D Numpy array for training containing the corresponding labels for each example
            X_val: a 2D Numpy array for validation where each row contains an example,
            padded by 1 column for the bias
            Y_val: a 1D Numpy array for validation containing the corresponding labels for each example
        @returns:
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
        '''
        train_errors = []
        val_errors = []
        #[TODO] train model and calculate train and validation errors here for each lambda
        for lbda in lambda_list:
            self.lmbda = lbda
            self.train(X_train, Y_train)
            train_errors.append([1 - self.accuracy(X_train, Y_train)])
            self.train(X_val, Y_val)
            val_errors.append([1 - self.accuracy(X_val, Y_val)])

        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset, k):
        '''
        Helper function for k-fold cross validation. Evenly split the indices of a
        dataset into k groups.

        For example, indices = [0, 1, 2, 3] with k = 2 may have an output
        indices_split = [[1, 3], [2, 0]].
        
        Please don't change this.
        @params:
            dataset: a Numpy array where each row contains an example
            k: an integer, which is the number of folds
        @return:
            indices_split: a list containing k groups of indices
        '''
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        indices_split = np.split(indices[:fold_size*k], k)
        return indices_split

    def runKFold(self, lambda_list, X, Y, k = 3):
        '''
        Run k-fold cross validation on X and Y with respect to each lambda. Return all k-fold
        errors.
        
        Each run of k-fold involves k iterations. For an arbitrary iteration i, the i-th fold is
        used as testing data while the rest k-1 folds are combined as one set of training data. The k results are
        averaged as the cross validation error.

        @params:
            lambda_list: a list of lambdas
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
            k: an integer, which is the number of folds, k is 3 by default
        @return:
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        '''
        k_fold_errors = []
        for lmbda in lambda_list:
            self.lmbda = lmbda
            #[TODO] call _kFoldSplitIndices to split indices into k groups randomly
            indices_split = self._kFoldSplitIndices(X, k)
            # indices_split_Y = self._kFoldSplitIndices(Y, k)

            #[TODO] for each iteration i = 1...k, train the model using lmbda
            # on k−1 folds of data. Then test with the i-th fold.
            errors = 0
            test_set = []
            test_set_Y = []
            train_set = []
            train_set_Y = []
            for i in range(k):
                for index in indices_split[i]:
                    test_set.append([X[index]])
                    test_set_Y.append([Y[index]])
                if i == 0:
                    indices = indices_split[i+1:]
                elif i == k-1:
                    indices = indices_split[:i]
                else:
                    indices = np.concatenate((indices_split[:i], indices_split[i+1:]), axis=0)
                for indexx in indices:
                    train_set.append([X[indexx]])
                    train_set_Y.append([Y[indexx]])
                self.train(np.asarray(train_set).reshape(-1, X.shape[1]), np.asarray(train_set_Y).flatten())
                errors += 1 - self.accuracy(np.squeeze(np.asarray(test_set)), np.squeeze(np.asarray(test_set_Y)))

            #[TODO] calculate and record the cross validation error by averaging total errors
            errors = errors/k
            k_fold_errors.append([errors])

        return k_fold_errors

    def plotError(self, lambda_list, train_errors, val_errors, k_fold_errors):
        '''
        Produce a plot of the cost function on the training and validation sets, and the
        cost function of k-fold with respect to the regularization parameter lambda. Use this plot
        to determine a valid lambda.
        @params:
            lambda_list: a list of lambdas
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        @return:
            None
        '''
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label = 'training error')
        plt.semilogx(lambda_list, val_errors, label = 'validation error')
        plt.semilogx(lambda_list, k_fold_errors, label = 'k-fold error')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.show()