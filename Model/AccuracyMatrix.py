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
    """

    def __init__(self):
        pass
    
    def cluster_UPGMA(ICMP_Message):
        """ Cluster PCAP Messages with UPGMA
        Returns a dictionary with the cluster, predicted labels and true labels 
        """
    
        cluster = Format.clusterByAlignment(ICMP_Message) #Output a list of Symbol Objects
        cluster_no = len(cluster) # num of clusters

        clusters_result = {} # dictionary that contains the cluster, predicted and truth values
        cluster_predicted = [] # list of all the message type from the inital dataset
        cluster_true = [] # list of all the message type from the inital dataset
        
        msg_per_cluster = [] # list to store num of message for each cluster

        # finding num of messages for each cluster
        for i in range(cluster_no):
            message_count = len(cluster[i].messages)
            msg_per_cluster.append(message_count)

        # Labelling the cluster type by the majority message type 
        for index,symbol in enumerate(cluster):
            # finding the true labels for predicted cluster
            for cluster_msg in range(msg_per_cluster[index]): # iterate through number of message in cluster
                message_type = symbol.messages[cluster_msg].l4MessageType 
                cluster_true.append(message_type)

            # finding the majorty element and populing the cluster predicted with the majority element
            majority_type = majority_element(cluster_true) # finding the majority element
            cluster_predicted = np.full((1,msg_per_cluster[index]),majority_type).tolist() # populing the cluster predicted with the majority element
            clusters_result[index] = cluster_predicted[0],cluster_true # stores into dictionary (cluster_predicted, cluster_true)

        y_predict,y_true = prepare_result(clusters_result)

        return y_predict,y_true

    def prepare_result(clusters_result):
        """ Converts results from dictionary to list form
        Returns y_pred and y_true in tuple 
        
        e.g 
        
        >>> cluster = AccuracyMatrix.cluster_UPGMA("./test/resources/pcaps/icmp_big.pcap",importLayer=4)
        >>> AccuracyMatrix.visualise(cluster) # to visualise confusion matrix
        """
    
        y_predict = []
        y_true = []
        for key,value in clusters_result.items():
            y_predict += value[0] # pred = value[0]
            y_true += value[1] # true = value[1]

        return y_predict,y_true

    def visualise_confusion(y_predict,y_true):
        """Returns a visualisation of the confusion matrix based on the result
        Lack Multi Cluster support"""
    
        # Obtaining the turth and predicted labels from dictionary
        y_pred = clusters_result[0]
        y_true = clusters_result[1] 

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
        plt.show();
