#-*- coding: utf-8 -*-

#+---------------------------------------------------------------------------+
#|          01001110 01100101 01110100 01111010 01101111 01100010            |
#|                                                                           |
#|               Netzob : Inferring communication protocols                  |
#+---------------------------------------------------------------------------+
#| Copyright (C) 2011-2017 Georges Bossert and Frédéric Guihéry              |
#| This program is free software: you can redistribute it and/or modify      |
#| it under the terms of the GNU General Public License as published by      |
#| the Free Software Foundation, either version 3 of the License, or         |
#| (at your option) any later version.                                       |
#|                                                                           |
#| This program is distributed in the hope that it will be useful,           |
#| but WITHOUT ANY WARRANTY; without even the implied warranty of            |
#| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              |
#| GNU General Public License for more details.                              |
#|                                                                           |
#| You should have received a copy of the GNU General Public License         |
#| along with this program. If not, see <http://www.gnu.org/licenses/>.      |
#+---------------------------------------------------------------------------+
#| @url      : http://www.netzob.org                                         |
#| @contact  : contact@netzob.org                                            |
#| @sponsors : Amossys, http://www.amossys.fr                                |
#|             Supélec, http://www.rennes.supelec.fr/ren/rd/cidre/           |
#+---------------------------------------------------------------------------+

#+---------------------------------------------------------------------------+
#| File contributors :                                                       |
#|       - Chia Yong Kang                |
#|                    |
#+---------------------------------------------------------------------------+

#+---------------------------------------------------------------------------+
#| Standard library imports                                                  |
#+---------------------------------------------------------------------------+

#+---------------------------------------------------------------------------+
#| Related third party imports                                               |
#+---------------------------------------------------------------------------+

#+---------------------------------------------------------------------------+
#| Local application imports                                                 |
#+---------------------------------------------------------------------------+

from netzob.all import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

class AccuracyMatrix:
    """This class implements methods to test the performace of different algorithm
    in clustering Network Messages
    
    >>> from netzob.Inference.Model import AccuracyMatrix
    >>> cluster = AccuracyMatrix.cluster_UPGMA("./test/resources/pcaps/icmp_big.pcap",importLayer=4).values()
    >>> print(cluster)
    
    
    """

    def __init__(self):
        pass
    
def cluster_UPGMA(message):
    """ Clusters and labels PCAP Messages with UPGMA for evaluation.
    
    Returns a tuple 
    with the cluster of predicted labels , true labels and no of message
    
    Output : (cluster_predicted,cluster_true,msg_per_cluster)
    """
    
    cluster = Format.clusterByAlignment(message) #Output a list of Symbol Objects
    cluster_no = len(cluster) # num of clusters
    print("No of clusters {}".format(cluster_no))

    clusters_result = {} # dictionary that contains the cluster, predicted and truth values
    cluster_predicted = [] # list of all the message type from the inital dataset
    cluster_true = [] # list of all the message type from the inital dataset
    
    msg_per_cluster = [] # list to store num of message for each cluster
    # finding num of messages for each cluster
    for i in range(cluster_no):
        message_count = len(cluster[i].messages)
        msg_per_cluster.append(message_count)

    # Labelling the each cluster type by the majority message type 
    for index,symbol in enumerate(cluster): # iterating through each cluster
        # finding the true labels for predicted cluster
        for cluster_msg in range(msg_per_cluster[index]): # iterate through number of message in  each cluster
            print(symbol.messages[cluster_msg].l4MessageType)
            print(cluster_msg)
            # For ICMP 
            if symbol.messages[cluster_msg].l4Protocol == 'ICMP':
                msg_protocol = 'ICMP' # Set the L4 Protocol. Different protocol has different method of labelling the true cluster.
                message_type = 'ICMP' + symbol.messages[cluster_msg].l4MessageType 
                cluster_true.append(message_type)
            # For TCP
            elif symbol.messages[cluster_msg].l4Protocol == 'TCP':
                msg_protocol = 'TCP' # Set the L4 Protocol. Different protocol has different method of labelling the true cluster.          
            # For UDP
            elif symbol.messages[cluster_msg].l4Protocol == 'UDP':
                msg_protocol = 'UDP'
                cluster_true.append(msg_protocol)

        # Labelling the predicted cluster
        cluster_majority = np.full((1,len(cluster[index].messages)),majority_element(cluster_true)).tolist() # finding the majority type of message per cluster
        cluster_predicted += cluster_majority # Adding to the list of predicted labels for cluster
             
    clusters_result = cluster_predicted[0],cluster_true # stores into dictionary (cluster_predicted, cluster_true)

    return cluster_predicted,cluster_true,msg_per_cluster
    
    def visualise_confusion(clusters_result):
        """Returns a visualisation of the confusion matrix based on the result

        e.g 
        >>> cluster = AccuracyMatrix.cluster_UPGMA("./test/resources/pcaps/icmp_big.pcap",importLayer=4)
        >>> AccuracyMatrix.visualise(cluster) # to visualise confusion matrix
        """
        
        # Obtaining the turth and predicted labels from dictionary
        y_pred = clusters_result[0]
        y_true = clusters_result[1] 
        
        unique_types = np.unique(np.array(y_true))# Finding the number of unique values from the true labels.
        unique_clusters = len(unique_types)# Number of unique clusters
        cluster_no = len(clusters_result[2]) # Number of clusters predicted
        accuracy = metrics.accuracy_score(y_true, y_pred) # Calculating accuracy score

        # Metrics
        print("Number of Message types : {}".format(unique_types))
        print("Number of Clusters : {}".format(unique_clusters))
        print("Number of Clusters predicted : {}".format(cluster_no))
        print("Percentage Accuracy in predicted cluster{:.2%} ".format(accuracy))

        # Creating a confusion matrix from y_true and y_pred
        cm = confusion_matrix(y_true,y_pred)
        exp_series = pd.Series(y_true)
        pred_series = pd.Series(y_pred)
        # return pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)
        plt.imshow(cm,cmap=plt.cm.Blues,interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix without Normalization')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        tick_marks = np.arange(len(set(y_true))) # length of classes
        class_labels = ['0','8']
        tick_marks
        plt.xticks(tick_marks,class_labels)
        plt.yticks(tick_marks,class_labels)
        
        # plotting text value inside cells
        thresh = cm.max() / 2.
        for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(cm[i,j],'d'),horizontalalignment='center',color='white' if cm[i,j] >thresh else 'black')
        plt.show(); # Plots the confusion matrix

    def majority_element(arr):
        """Returns the majority value in the array.
        Implemented using Boyer–Moore majority vote algorithm"""

        counter, possible_element = 0, None
        for i in arr:
            if counter == 0:
                possible_element, counter = i, 1
            elif i == possible_element:
                counter += 1
            else:
                counter -= 1

        return possible_element
