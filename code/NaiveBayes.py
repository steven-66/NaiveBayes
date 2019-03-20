import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time

from imdb import IMDBdata
import matplotlib.pyplot as plt
class NaiveBayes:
    def  __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.vocab.GetVocabSize()
        self.count_positive =[]
        self.count_negative =[]
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.5
        self.P_negative = 0.5
        self.deno_pos = 1.0
        self.deno_neg = 1.0
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()

        self.num_positive_reviews = 0
        self.num_negative_reviews = 0

        self.count_positive = np.zeros([1,X.shape[1]])
        self.count_negative = np.zeros([1,X.shape[1]])

        self.total_positive_words = 0
        self.total_negative_words = 0

        for i in positive_indices:
            # this is a positive file
            self.num_positive_reviews +=1
            #count words in this positive file which value is wordid
            tmp = np.nonzero(X[i])
            self.total_positive_words += np.sum(X[tmp])
            self.count_positive[0,tmp[1]] = np.add(self.count_positive[0,tmp[1]],X[i][tmp])
        for i in negative_indices:
            self.num_negative_reviews +=1
            # count words from  a negative file
            tmp = np.nonzero(X[i])
            self.total_negative_words += np.sum(X[tmp])
            self.count_negative[0,tmp[1]] = np.add(self.count_negative[0,tmp[1]],X[i][tmp])



        self.deno_pos = 1.0
        self.deno_neg = 1.0

        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X, probThresh):
        #TODO: Implement Naive Bayes Classification
        self.P_positive = 0
        self.P_negative = 0
        pred_labels = []
        self.P_negative= self.num_negative_reviews / (self.num_positive_reviews+ self.num_negative_reviews) # P(Y=1)
        self.P_positive= self.num_positive_reviews / (self.num_positive_reviews+ self.num_negative_reviews) # P(Y=-1)
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero() # look at each file
            log_sum_positive = 0 # sum each positive feature
            log_sum_negative =0 # sum each negative feature
            for j in range(len(z[0])):
                # Look at each feature
                word_counts_positive = 0
                word_counts_negative = 0
                if z[1][j]< self.data.X.shape[1]: # if testing word in training data.vocab
                    word_counts_positive = self.count_positive[0, z[1][j]]
                    word_counts_negative = self.count_negative[0, z[1][j]]
                # do laplacian smoothing
                positive_feature = (word_counts_positive + self.ALPHA) / (self.total_positive_words + self.vocab_len)
                log_sum_positive += log(positive_feature)
                negative_feature = (word_counts_negative + self.ALPHA) / (self.total_positive_words + self.vocab_len)
                log_sum_negative += log(negative_feature)
            deno_all = self.LogSum(log_sum_negative, log_sum_positive) # deno = log(P(X|Y=1)+P(X|Y=-1)
            score2positve = exp(log_sum_positive - deno_all) # get probability to be positive
            score2negative = exp(log_sum_negative - deno_all)
            if score2positve > score2negative:            # Predict positive when plot PR-curve, change to score2positive > probThresh
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)

        return pred_labels

    def LogSum(self, logx, logy):
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):

        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            z = np.nonzero(test.X[i])
            log_sum_positive = 0
            log_sum_negative = 0

            for j in range(len(z[0])):
                # Look at each feature
                word_counts_positive = 0
                word_counts_negative = 0
                if z[1][j] < self.data.X.shape[1]:  # if testing word in training data.vocab
                    word_counts_positive = self.count_positive[0, z[1][j]]
                    word_counts_negative = self.count_negative[0, z[1][j]]
                # laplacian smoothing
                positive_feature = (word_counts_positive + self.ALPHA) / (self.total_positive_words + self.vocab_len)
                log_sum_positive += log(positive_feature)
                negative_feature = (word_counts_negative + self.ALPHA) / (self.total_positive_words + self.vocab_len)
                log_sum_negative += log(negative_feature)
            deno = self.LogSum(log_sum_negative, log_sum_positive) # deno = log(P(X|Y=1)+P(X|Y=-1)

            predicted_prob_positive = exp(log_sum_positive - deno)
            predicted_prob_negative = exp(log_sum_negative - deno)

            if predicted_prob_positive > predicted_prob_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0

            # print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print('actual_label:',test.Y[i],'predicted_label:', predicted_label, 'predicted_prob_positive:',predicted_prob_positive, 'predicted_prob_negative:',predicted_prob_negative)

    # Evaluate performance on test data
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X,1.0)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()
    # caculate the polarity of each words find most positive and negative words
    # use log-odd-ratio as the weight for each word
    def get_word_polarity(self):
        pos_word = (self.count_positive + self.ALPHA) / (self.total_positive_words + self.vocab_len)
        neg_word = (self.count_negative + self.ALPHA) / (self.total_negative_words + self.vocab_len)
        word_polarity = np.log(pos_word/neg_word)
        neg_index= np.argsort(word_polarity)[0][:20]    # get the first 20 positive words
        pos_index= np.argsort(word_polarity)[0][::-1][:20] # get the first 20 negative words

        print('20 most positive word:\n')
        for i in range(20):
            word = self.data.vocab.GetWord(pos_index[i])
            weight = word_polarity[0,pos_index[i]]
            print(word, weight)
        print('20 most negative word:\n')
        for i in range(20):
            word = self.data.vocab.GetWord(neg_index[i])
            weight = word_polarity[0,neg_index[i]]
            print(word,weight)

    # param: TP means correctly classified positive sample
    # FP means a negative sample classified into positive mistakenly
    # FN means a positive sample predicted into negative mistakenly
    # TN means a negative sample predicted into negative correctly
    def EvalPrecision(self,TP,TN,FN,FP):
        return TP / (TP + FP), TN / (TN + FN)

    def EvalRecall(self,TP,TN,FN,FP):
        return TP / (TP + FN), TN / (TN + FP)

    def Plot_precision_recall(self, threshholds, test):
        pos_precision = [] # precision for positive class
        neg_precision = [] # precision for negative class
        pos_recall = [] # recall for positive class
        neg_recall = [] # recall for negative class
        for i in threshholds:
            Y_pred = self.PredictLabel(test.X, i)
            sum = Y_pred + test.Y
            sub = Y_pred - test.Y
            TP = len(np.argwhere(sum == 2.0).flatten())
            TN = len(np.argwhere(sum == -2.0).flatten())
            FP = len(np.argwhere(sub == 2.0).flatten())
            FN = len(np.argwhere(sub == -2.0).flatten())
            p, n = self.EvalPrecision(TP,TN,FN,FP)
            pos_precision.append(p)
            neg_precision.append(n)
            p, n = self.EvalRecall(TP,TN,FN,FP)
            pos_recall.append(p)
            neg_recall.append(n)
        # plot PR-curve for positive class
        fig=plt.figure()
        ax1= fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_xlabel('positive-recall')
        ax1.set_ylabel('positive-precision')
        ax1.plot(pos_recall,pos_precision)
        # plot PR-curve for negative class
        ax2.set_xlabel('negative-recall')
        ax2.set_ylabel('negative-precision')
        ax2.plot(neg_recall,neg_precision)
        plt.show()
if __name__ == "__main__":

    # t=np.zeros([1,3])
    # print(len(t[0]))
    # t=np.argsort(t[0])[::-1][:2]
    # print(t)
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    traindata.vocab.locked = False  # New Line
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    # nb.Plot_precision_recall([ 0.1, 0.35,0.4,0.5,0.6,0.65,0.7,0.8],testdata) # plot PR-curve
    nb.PredictProb(testdata,range(10)) # predict probality of first 10 file
    nb.get_word_polarity()

