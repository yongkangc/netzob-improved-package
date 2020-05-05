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
from netzob.all import *
from netzob.Model.performance_matrix import PerformanceMatrix

from pprint import pprint
import itertools

# Data Analytics mods
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


# NLP Modules
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.test.utils import datapath, get_tmpfile
import pyLDAvis.gensim
import warnings
from gensim.models import TfidfModel
from gensim.similarities import Similarity


class LDAModel:
    """
    This Class implements methods for
    - N-gram
    - TF/IDF
    - LDA
    - PCA
    - Kmeans

    to cluster the messages

    >>> from netzob.Model.lda_kmeans import LDAModel
    >>> model = LDAModel()
    >>> model.clusterByLDA(file_path ="./test/resources/pcaps/utf8-encoded-messages.pcap", import_layer = 4)
    >>> print(cluster)

    """

    def __init__(self):
        pass

    def clusterByLDA(self,num_topics, num_clusters, *args):
        """ Clusters the message type using Latent Dirichlet Allocation
        *args stands for the filepath that the message is imported from

        Returns LDA Topic distribution
        """
        message = self.import_multiple_message(*args)
        msg_type = self.identify_msg_type(message)
        print("\n" + "MSG Type : " + str(msg_type) + "\n")
        true_dict = self.count_element(msg_type)

        # generate a text document of message (located in data.txt)
        self.write_message(message)

        # Processing the text document (located in data.txt)
        docs = self.msg_to_bytes()
        dictionary = self.create_dict(docs)
        corpus = self.create_corpus(docs)

        # Set training parameters for LDA
        num_topics = num_topics
        chunksize = 1  # how many documents are processed at a time
        passes = 30  # how often we train the model on the entire corpus.
        iterations = 1000
        eval_every = 1  # For logging
        minimum_probability = 0.0

        # Set parameter for Kmeans
        num_clusters = num_clusters

        # Make a index to word dictionary.
        temp = dictionary[0]  # initialize the dictionary
        id2word = dictionary.id2token

        # Train the model on the corpus.
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
        )

        # # Train a multicore LDA model
        # lda_model = LdaMulticore(
        #     corpus=corpus,
        #     id2word=id2word,
        #     chunksize=chunksize,
        #     alpha='auto',
        #     eta='auto',
        #     iterations=iterations,
        #     num_topics=num_topics,
        #     passes=passes,
        #     eval_every=eval_every,
        #     minimum_probability=0.0,
        #     workers=1,
        # )
        temp_file = datapath("model_lda")
        lda_model.save(temp_file)  # saving the model in "tempfile"

        top_topics = lda_model.top_topics(corpus)
        # Get topic distribution and forms a list
        topic_dist = [lda_model.get_document_topics(item, minimum_probability=0.0) for item in corpus]


        # visualise_LDA(lda_model, corpus, dictionary)

        # outputs results in a txt file
        self.write_result(lda_model, avg_topic_coherence, topic_dist)

        # Clustering by Kmeans
        clusters_result = self.clusterByKMeans(num_topics=num_topics, num_clusters=num_clusters, lda_model=lda_model,
                                          msg_type=msg_type, docs=docs, true_dict=true_dict)

        # Getting the Performance matrix (accuracy, confusion matrix, precision)
        acc = self.visualise_confusion(clusters_result=clusters_result)

    def clusterByKMeans(self,num_topics, num_clusters, lda_model, msg_type, docs, true_dict):
        """ Clusters the output of LDA by Kmeans
        Returns y true and y predicted as well. And Kmeans visualisation."""
        corpus = self.create_corpus(docs)
        topic_dist = [lda_model.get_document_topics(item, minimum_probability=0.0) for item in corpus]
        X = pd.DataFrame(topic_dist)  # Dataframe of the result. Use Jupyter notebook to view.

        entry_num = 1  # index one -> the probability of message in topics

        # Removing the id from the tuple, leaving the probablity of each word being in topic
        for row in X.iterrows():
            for i in range(0, num_topics):
                row[entry_num][i] = row[entry_num][i][1]

        print(len(X.index))
        # PCA first to reduce dimensionality for visualisation
        pca = PCA(n_components=2)
        PC = pca.fit_transform(X)

        # Applying Kmeans to get labels(cluster no)
        num_clusters = num_clusters  # need to write a function to find out the optimal numb of topics
        kmeans = KMeans(n_clusters=num_clusters, n_init=10).fit_predict(PC)

        # Dataframe with labels
        Y = pd.DataFrame()
        Y["true_labels"] = msg_type
        cluster_predicted = kmeans.tolist()

        print("cluster predicted" + str(len(cluster_predicted)))
        print(cluster_predicted)
        print("cluster predicted" + str(len(msg_type)))

        Y["pred_labels"] = cluster_predicted
        Y.groupby("pred_labels")
        clustered_labels = {}
        for (i, row) in Y.iterrows():
            if row["pred_labels"] in clustered_labels:
                clustered_labels[row["pred_labels"]].append(row["true_labels"])
            else:
                clustered_labels[row["pred_labels"]] = [row["true_labels"]]

        y_pred = []
        for i in clustered_labels:
            # Labelling the predicted cluster
            pred_dict = PerformanceMatrix.count_element(clustered_labels[i])
            maj = PerformanceMatrix.normalise_pred(clustered_labels[i], true_dict, pred_dict)
            # maj = majority_element(clustered_labels[i])
            cluster_maj = [maj for i in range(len(clustered_labels[i]))]
            # print(cluster_predicted)
            y_pred.extend(cluster_maj)  # Adding to the list of predicted labels for cluster

        y_true = []
        for i in clustered_labels:
            y_true.extend(clustered_labels[i])

        # Plotting Kmeans
        # iterating through no of categories
        print("clustered_labels")
        print(clustered_labels)
        print("\n")
        print("y_true : ")
        print(y_true)
        print("\n")
        print("y_pred: ")
        print(y_pred)
        print("\n")

        fig, ax = plt.subplots()
        for i in np.unique(kmeans):
            plotx = []
            ploty = []
            for j in range(PC.shape[0]):
                if kmeans[j] == i:
                    plotx.append(PC[j][0])
                    ploty.append(PC[j][1])

            # Plotting the graph
            plt.scatter(plotx, ploty, label=i)  # projected points to the axis

        # plt.ylim([1e-7,1.5e-7])
        # plt.xlim([-1e-6,1e-6])
        ax.legend()
        plt.show()

        return (y_pred, y_true, np.unique(kmeans), num_topics)

    def tf_idf(self, corpus):
        """Using TF/IDF to vectorize the data
        Returns tfidf weighted corpus"""
        tfidf = TfidfModel(corpus)  # fit model
        # tfidf = [model[corpus[i]] for i in range(len(corpus))]
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    def identify_msg_type(self, message):
        """" Returns Message Protocol and corresponding Message type """
        # symbol = Symbol(messages=message)

        msg_type = []
        for i in range(len(message)):
            if message[i].l4Protocol == 'TCP':
                type = str(message[i].l4MessageType)
                msg_type.append(type)
            elif message[i].l4Protocol == 'ICMP':
                msg_protocol = 'ICMP'  # Set the L4 Protocol. Different protocol has different method of labelling the true cluster.
                msg_type.append('ICMP' + message[i].l4MessageType)
            elif message[i].l4Protocol == 'UDP':
                msg_protocol = 'UDP'
                msg_type.append(msg_protocol)

        return msg_type

    def visualise_LDA(self, lda_model, corpus, dictionary):
        """Visualise the LDA results"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(visualisation, 'LDA_Visualisation.html')

    def write_result(self, lda_model, avg_topic_coherence, topic_dist):
        """Create a text document of the result"""
        with open("result4.txt", "w") as f:
            # pprint(topic_dist, stream=f)
            # print(topic_dist, file=f)
            print('Average topic coherence: %.4f.' % avg_topic_coherence, file=f)

    def load_lda(self):
        """Loading the LDA model from save file
        Returns the LDA Model"""
        corpus = self.create_corpus()
        temp_file = datapath("model")
        lda_model = gensim.models.LdaModel.load(temp_file)

        return lda_model

    def import_message(self,file_path,importLayer=4):
        """Abstracted function to import messages"""
        message = PCAPImporter.readFile(file_path, importLayer=importLayer).values()
        return message

    def import_multiple_message(self,importLayer=4, *args):
        """Abstracted function to import multiple messages"""

        for count, elem in enumerate(args):
            message_single = PCAPImporter.readFile(elem, importLayer=importLayer).values()
            if count < 1:
                message_all = message_single
            else:
                message_all += message_single

        return message_all

    def write_message(self, message):
        """Outputs the message in data.txt
        For LDA to process """
        datafile = open("data.txt", "w")

        from netzob.all import Symbol
        symbol = Symbol(messages=message)
        # For Symbol
        datafile.writelines(str(symbol) + "\n")

        datafile.close()


    def parse_input(self, text):
        return text.strip("\n").strip(" ").strip("b")

    def tokenize_hex(self, text):
        # re.split(r'\\x'+'\\',text)
        return text.split("\\")

    def is_hex(self, text):
        return text != "\'"

    def parse_hex(self, text):
        return text.strip("x")

    def msg_to_bytes(self):
        """ Breaking Messages into bytes
        Returns a list of Messages
        """
        f = open("/content/drive/My Drive/DSO Presentation/dataset/tcp_icmp_udp.txt", "r")
        # f = open("/content/drive/My Drive/DSO Presentation/dataset/TCP_ICMP_UDP_HTTP.txt", "r")
        # print(sent_tokenize(text))
        text = f.readlines()
        doc = []
        for line in text:
            parsed_hex = []
            if "\\x" in line:
                line = self.parse_input(line)
                tokenized_hex = self.tokenize_hex(line)
                for token in tokenized_hex:
                    if self.is_hex(token):
                        parsed_hex.append(self.parse_hex(token))

                # limiting the header to 70 bytes
                # doc.append(header_lim(parsed_hex))
                doc.append((parsed_hex))

            elif any(x in line for x in ["GET", "HTTP"]):
                line = self.parse_input(line)
                # tokenized_hex = tokenize_hex(line)
                # tokenized_hex = tokenize_ascii(line)
                for token in line:
                    if self.is_hex(token):
                        if not any(substring in token for substring in [" ", "'", "\r", "\n", "", "r", "n"]):
                            parsed_hex.append(self.parse_hex(token))
                doc.append(self.header_lim(parsed_hex))

        return doc

    def n_gram_message(self, docs):
        # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
        bigram = Phrases(docs, min_count=10)
        trigram = Phrases(bigram[docs])

        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
            for token in trigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
        return docs

    def filter_tokens(self, dictionary):
        """ Filter out words that occur less than "no_below" documents, or more than "no_above" of the documents.
        Returns dictionary with filtered tokens"""
        no_below = 10
        no_above = 0.2
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)

        return dictionary

    def create_dict(self, docs):
        """ Create a dictionary representation of the documents."""
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)
        return dictionary

    def create_corpus(self, docs):
        """Returns a TF/IDF Weighted corpus"""
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)
        # Create a dictionary representation of the documents.
        # Bag-of-words representation of the documents.
        corpus = [dictionary.doc2bow(doc) for doc in docs]  # output (ID:frequency)
        # Using Tf-Idf
        corpus_tfidf = self.tf_idf(corpus)  # Gensim object
        return corpus_tfidf

    def similarity_matrix(self,corpus, dictionary):
        """Compute cosine similarity against a corpus of documents by storing the index matrix in memory."""
        # index = MatrixSimilarity(corpus, num_features=len(dictionary))
        index_temp = get_tmpfile("index")
        index = Similarity(index_temp, corpus, num_features=len(dictionary))  # create index
        for sims in index[corpus]:
            pprint(sims)
