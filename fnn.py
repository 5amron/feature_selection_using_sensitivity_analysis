import numpy as np
np.set_printoptions(precision=4, formatter={'float_kind':'{:f}'.format})
import math
import random
import copy
import os





class FnnSingleLayer(object):
    def __init__(self, hidd_num, print_every):
        self.hidd_num = hidd_num
        self.print_every = print_every



    def forward(self, x):
        """ forward """
        LL0 = np.atleast_2d(x)                    # (1,8)
        LL1 = self.sigmoid(np.dot(LL0, self.W0))  # (1,12) = (1, 8).(8,12)
        LL2 = np.atleast_2d(self.stablesoftmax(np.dot(LL1, self.W1)))  # (1, 10) = (1,12).(12,10)

        return LL0, LL1, LL2




    def backprop(self, L0, L1, L2, n):
        acctual_y = self.y_data[n]

        """ backpropogation """
        correct_outs = np.zeros(self.class_num, dtype="int64").reshape(1, self.class_num)
        correct_outs[0, acctual_y] = 1

        L2_error = correct_outs - L2 # (1,10)-(1,10)
        L2_delta = L2_error*self.learn_rate

        L1_error = np.dot(L2_delta, self.W1.T) # (1,12) = (1,10).(10,12)
        L1_delta = np.multiply(L1_error, self.sigmoid_dir(L1))*self.learn_rate

        self.W1 += np.dot(np.atleast_2d(L1).T, np.atleast_2d(L2_delta))
        self.W0 += np.dot(np.atleast_2d(L0).T, np.atleast_2d(L1_delta))



    def prune_input_nodes(self, new_x, new_x_val, new_x_test, nodes):
        self.x_data = new_x
        self.x_test_set = new_x_test
        self.x_val_set = new_x_val
        self.W0 = np.delete(self.W0, nodes, 0)


    def cross_entropy_error(self, x, y):
        this_set_sam_num = x.shape[0]
        total = 0
        for n in range(this_set_sam_num):
            (L0, L1, L2) = self.forward(x[n])
            acctual_y = y[n]
            total = total - np.log(L2[0, acctual_y])

        final = total / this_set_sam_num

        return final[0]




    def predict(self, x):
        (L0, L1, L2) = self.forward(x)
        # probs = L2 / np.sum(L2, axis=1, keepdims=True) # axis=1 means sum for cols
        # try:
        return np.where(L2 == np.max(L2))[1][0]
        # except:
        #     print(L2)

    def calc_acc(self, x, y):
        acc_count = 0
        ##### be careful about minus one here (handled it in datasets theirselves)
        for k in range(x.shape[0]):
            if (self.predict(x[k]) == y[k]):
                acc_count += 1

        acc = acc_count/x.shape[0]

        return acc




    def print_stuff(self, e):
        if(self.dontshow):
            return
        if(e % self.print_every == 0):
            cee_train = self.cross_entropy_error(self.x_data, self.y_data)
            cee_val = self.cross_entropy_error(self.x_val_set, self.y_val_set)
            train_acc = self.calc_acc(self.x_data, self.y_data)
            val_acc = self.calc_acc(self.x_val_set, self.y_val_set)
            test_acc = self.calc_acc(self.x_test_set, self.y_test_set)
            self._our_log_("---------- e="+str(e)+" ----------")
            self._our_log_("cee_train : " + str(cee_train) + " ({0:.2f})".format(train_acc*100) + "%")
            self._our_log_("cur cee_val : " + str(cee_val) + " ({0:.2f})".format(val_acc*100) + "%")
            self._our_log_("best_val_cee_sofar : " + str(self.best_val_cee_sofar) + " ({0:.2f})".format(self.best_val_cee_sofar_acc*100) + "%")
            self._our_log_("cur test acc : " + "({0:.2f})".format(test_acc*100) + "%")





    def train(self, x_data, y_data, x_val_set, y_val_set, x_test_set, y_test_set, epochs, learn_rate=0.1, patience=20, new=True, dontshow=False):
        np.random.seed(1)
        self.x_data = x_data
        self.y_data = y_data
        self.x_test_set = x_test_set
        self.y_test_set = y_test_set
        self.x_val_set = x_val_set
        self.y_val_set = y_val_set
        self.train_sam_num = self.x_data.shape[0]-1
        self.feature_num = self.x_data.shape[1]
        self.class_num = len(np.unique(self.y_data))
        self.learn_rate = learn_rate
        self.dontshow = dontshow
        self.patience = patience
        if(new):
            self.B1 = np.zeros((1, self.hidd_num))
            self.B2 = np.zeros((1, self.class_num))
            self.W0 = 2*np.random.random((self.feature_num, self.hidd_num))-1
            self.W1 = 2*np.random.random((self.hidd_num, self.class_num))-1
            self.best_val_cee_sofar = 100000
            self.p_count = 0
            self.flagg = False


        self.check_perform()
        self.print_stuff(0)
        for e in range(1, epochs+1):
            if(self.p_count > self.patience): # breaking before another epoch!
                self.p_count = 0
                self._our_log_("eeeeeeearly stooooopping")
                break

            for n in range(self.train_sam_num):

                (L0, L1, L2) = self.forward(self.x_data[n])
                self.backprop(L0, L1, L2, n)

            self.check_perform()
            self.print_stuff(e)
            self.last_e = e



        self.be_the_best()
        acc_test = self.calc_acc(self.x_test_set, self.y_test_set)
        cee_train = self.cross_entropy_error(self.x_data, self.y_data)
        cee_val = self.cross_entropy_error(self.x_val_set, self.y_val_set)
        acc_val = self.calc_acc(self.x_val_set, self.y_val_set)
        self._our_log_("++++++++++++ final ++++++++++++")
        self._our_log_("e : " + str(e))
        self._our_log_("cee_train : " + str(cee_train))
        self._our_log_("cur cee_val : " + str(cee_val))
        self._our_log_("best_val_cee_sofar : " + str(self.best_val_cee_sofar))
        self._our_log_("cur val acc : " + "({0:.2f})".format(acc_val*100) + "%")
        self._our_log_("cur test acc : " + "({0:.2f})".format(acc_test*100) + "%")
        self._our_log_("+++++++++++++++++++++++++++++++")



    @staticmethod
    def _our_log_(thing):
        print(thing)



    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def sigmoid_dir(x):
        return x*(1 - x)



    @staticmethod
    def stablesoftmax(x):
        """Compute the softmax of vector x in a numerically stable way."""
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
