# -*- coding: utf-8 -*-

# +---------------------------------------------------------------------------+
# |          01001110 01100101 01110100 01111010 01101111 01100010            |
# |                                                                           |
# |               Netzob : Inferring communication protocols                  |
# +---------------------------------------------------------------------------+
# | Copyright (C) 2011-2017 Georges Bossert and Frédéric Guihéry              |
# | This program is free software: you can redistribute it and/or modify      |
# | it under the terms of the GNU General Public License as published by      |
# | the Free Software Foundation, either version 3 of the License, or         |
# | (at your option) any later version.                                       |
# |                                                                           |``
# | This program is distributed in the hope that it will be useful,           |
# | but WITHOUT ANY WARRANTY; without even the implied warranty of            |
# | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the              |
# | GNU General Public License for more details.                              |
# |                                                                           |
# | You should have received a copy of the GNU General Public License         |
# | along with this program. If not, see <http://www.gnu.org/licenses/>.      |
# +---------------------------------------------------------------------------+
# | @url      : http://www.netzob.org                                         |
# | @contact  : contact@netzob.org                                            |
# | @sponsors : Amossys, http://www.amossys.fr                                |
# |             Supélec, http://www.rennes.supelec.fr/ren/rd/cidre/           |
# +---------------------------------------------------------------------------+

# +---------------------------------------------------------------------------+
# | File contributors :                                                       |
# |       - Chia Yong Kang  (ExtremelySunnyYk@github.io)                      |
# |
# +---------------------------------------------------------------------------+

# +---------------------------------------------------------------------------+
# | Standard library imports                                                  |
# +---------------------------------------------------------------------------+

# +---------------------------------------------------------------------------+
# | Related third party imports                                               |
# +---------------------------------------------------------------------------+

# +---------------------------------------------------------------------------+
# | Local application imports                                                 |
# +---------------------------------------------------------------------------+

from sklearn.metrics import confusion_matrix, precision_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from collections import Counter


class PerformanceMatrix:
    """This class implements methods to test the performace of different algorithm
    in clustering Network Messages
    
    >>> from netzob.Model.performance_matrix import PerformanceMatrix
    >>> cluster = cluster_UPGMA("./test/resources/pcaps/icmp_big.pcap",importLayer=4).values()
    >>> print(cluster)
    
    
    """

    def __init__(self):
        pass

    @staticmethod
    def cluster_UPGMA(message):
        """ Clusters and labels PCAP Messages with UPGMA for evaluation.
        
        Returns a tuple 
        with the cluster of predicted labels , true labels and no of message
        
        Output : (cluster_predicted,cluster_true,msg_per_cluster)
        """
        from netzob.all import Format

        msg_type = PerformanceMatrix.identify_msg_type(message)
        true_dict = PerformanceMatrix.count_element(msg_type)
        cluster = Format.clusterByAlignment(message)  # Output a list of Symbol Objects
        cluster_no = len(cluster)  # num of clusters

        cluster_predicted = []  # list of all the message type from the inital dataset
        cluster_true = []  # list of all the message type from the inital dataset



        msg_per_cluster = []  # list to store num of message for each cluster
        # finding num of messages for each cluster
        for i in range(cluster_no):
            message_count = len(cluster[i].messages)
            msg_per_cluster.append(message_count)

        # Labelling the each cluster type by the majority message type 
        for index, symbol in enumerate(cluster):  # iterating through each cluster
            # finding the true labels for predicted cluster
            for cluster_msg in range(msg_per_cluster[index]):  # iterate through number of message in  each cluster
                # For ICMP
                if symbol.messages[cluster_msg].l4Protocol == 'ICMP':
                    msg_protocol = 'ICMP'  # Set the L4 Protocol. Different protocol has different method of labelling the true cluster.
                    message_type = 'ICMP : ' + symbol.messages[cluster_msg].l4MessageType
                    cluster_true.append(msg_protocol)
                # For TCP
                elif symbol.messages[cluster_msg].l4Protocol == 'TCP':
                    if symbol.messages[cluster_msg].l4MessageType == "IRC":
                        msg_protocol = "IRC"
                    else:
                        msg_protocol = 'TCP'
                    # message_type = symbol.messages[
                    #     cluster_msg].l4MessageType  # Set the L4 Protocol. Different protocol has different method of labelling the true cluster.
                    cluster_true.append(msg_protocol)
                    # For UDP
                elif symbol.messages[cluster_msg].l4Protocol == 'UDP':
                    msg_protocol = 'UDP'
                    cluster_true.append(msg_protocol)

            # Getting the total number of types in each cluster
            pred_dict = PerformanceMatrix.count_element(cluster_true)
            print("pred_dict {}".format(pred_dict))
            print("true dict {}".format(true_dict))

            # finding the majority type of message per cluster
            # maj = PerformanceMatrix.majority_element(cluster_true)

            # finding the majority type with normalization
            maj = PerformanceMatrix.normalise_pred(cluster_true, true_dict,pred_dict)
            cluster_majority = [maj for i in range(len(cluster[index].messages))]

            print(cluster_majority)

            # Adding to the list of predicted labels for all cluster
            cluster_predicted.extend(cluster_majority)

        PerformanceMatrix.visualise_confusion([cluster_predicted, cluster_true, msg_per_cluster])

    @staticmethod
    def visualise_confusion(clusters_result):
        """Returns a visualisation of the confusion matrix based on the result"""

        # Obtaining the turth and predicted labels from dictionary
        y_pred = clusters_result[0]
        y_true = clusters_result[1]

        # Finding the unique values in the truth. This will tell us number of unique clusters
        unique_types_true = np.unique(np.array(y_true))
        unique_clusters_true = len(unique_types_true)  # Number of true unique clusters
        cluster_no = len(clusters_result[2])  # Number of clusters predicted
        accuracy = accuracy_score(y_true, y_pred)  # Calculating accuracy score

        print("Number of Message types : {}".format(unique_types_true))
        print("Number of Clusters : {}".format(unique_clusters_true))
        print("Number of Clusters predicted : {}".format(cluster_no))
        print("Percentage Accuracy in predicted cluster : {:.2%} ".format(accuracy))

        class_labels = list(set(y_true))  # Creating a list of unqiue labels
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)  # Creating a confusion matrix from y_true and y_pred

        # Calculating precision and recall
        # Using micro average as there might be a class imbalance (i.e more examples of one class than another)
        metric_score_micro = precision_recall_fscore_support(y_true, y_pred, average="micro")
        print("Precision Score is {:.2f}".format(metric_score_micro[0]))
        print("Recall Score is {:.2f}".format(metric_score_micro[1]))
        print("F Score is {:.2f}".format(metric_score_micro[2]))

        plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix without Normalization')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        tick_marks = np.arange(len(set(y_true)))  # length of classes

        # tick_marks
        plt.xticks(tick_marks, class_labels, fontsize=6)
        plt.yticks(tick_marks, class_labels, fontsize=7)

        # plotting text value inside cells
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.show()  # Plots the confusion matrix

    @staticmethod
    def import_message(self,file_path,importLayer=4):
        """Abstracted function to import messages"""
        message = PCAPImporter.readFile(file_path, importLayer=importLayer).values()
        return message

    @staticmethod
    def import_multiple_message(*args):
        """Abstracted function to import multiple messages"""

        for count, elem in enumerate(args):
            if elem == "./test/resources/pcaps/test_import_http.pcap":
                message_single = PCAPImporter.readFile(elem, importLayer=5).values()
                if count < 1:
                    message_all = message_single
                else:
                    message_all += message_single
            else:
                message_single = PCAPImporter.readFile(elem, importLayer=4).values()
                if count < 1:
                    message_all = message_single
                else:
                    message_all += message_single

        return message_all


    @staticmethod
    def majority_element(arr):
        """Returns the majority value in the array.
        Implemented using Boyer–Moore majority vote algorithm

        >>>  print(PerformanceMatrix.majority_element([1,2,3,1,12]))
        >>> 12


        """

        counter, possible_element = 0, None
        for i in arr:
            if counter == 0:
                possible_element, counter = i, 1
            elif i == possible_element:
                counter += 1
            else:
                counter -= 1

        return possible_element

    @staticmethod
    def normalise_pred(arr, true_dict, pred_dict):
        """ Finding the weighted average of the message type
        Returns the highest probability message type.
        """

        fraction_array = []
        for i in arr:
            if i in true_dict:
                fraction = pred_dict[i] / true_dict[i]
                fraction_array.append(fraction)
            else:
                print("no similarities for {}".format(i))
        print(fraction_array)
        index, value = max(enumerate(fraction_array), key=operator.itemgetter(1))

        return arr[index]

    @staticmethod
    def count_element(array):
        """Counts the unique message types in list
         Returns Dictionary of type : times
         """
        unique_elements = list(Counter(array).keys())
        element_frequency = list(Counter(array).values())

        dict = {}

        for index, key in enumerate(unique_elements):
            dict[key] = element_frequency[index]

        return dict

    @staticmethod
    def identify_msg_type(message):
        """" Returns Message Protocol and corresponding Message type """
        # symbol = Symbol(messages=message)

        msg_type = []
        for i in range(len(message)):

            if message[i].l4Protocol == 'TCP':
                # msg_protocol = str(message[i].l4MessageType)
                if message[i].l4MessageType == "IRC":
                    msg_protocol = "IRC"
                else:
                    msg_protocol = 'TCP'
                    # msg_protocol = message[i].l4MessageType

                msg_type.append(msg_protocol)
            elif message[i].l4Protocol == 'ICMP':
                msg_protocol = 'ICMP'  # Set the L4 Protocol. Different protocol has different method of labelling the true cluster.
                # msg_protocol = 'ICMP : ' + message[i].l4MessageType
                msg_type.append(msg_protocol)
            elif message[i].l4Protocol == 'UDP':
                msg_protocol = 'UDP'
                msg_type.append(msg_protocol)

        return msg_type


