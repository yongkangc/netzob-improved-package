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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from netzob.Inference.Model import PerformanceMatrix


class LDAModel:
    """
    This Class implements methods for
    - N-gram
    - TF/IDF
    - LDA
    - PCA
    - Kmeans

    to cluster the messages

    >>> from netzob.Inference.Model import LDAModel
    >>> cluster = LDAModel.clusterByLDA("./test/resources/pcaps/utf8-encoded-messages.pcap", 4)
    >>> print(cluster)

    """

    def __init__(self):
        pass

    def clusterByLDA(file_path, import_layer):
        """ Clusters the message type using Latent Dirichlet Allocation"""
        message = import_message(file_path, import_layer)
        msg_type = identify_msg_type(message)

        # generate a text document of message (located in data.txt)
        write_message(message)

        # Processing the text document (located in data.txt)
        docs = msg_to_bytes()
        dictionary = create_dict(docs)
        corpus = create_corpus(docs)

        # Set training parameters.
        num_topics = 8
        chunksize = 1  # how many documents are processed at a time
        passes = 30  # how often we train the model on the entire corpus.
        iterations = 500
        eval_every = 1  # For logging
        minimum_probability = 0.0

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
        temp_file = datapath("model")
        lda_model.save(temp_file)  # saving the model in "tempfile"

        top_topics = lda_model.top_topics(corpus)
        # Get topic distribution and forms a list
        topic_dist = [lda_model.get_document_topics(item, minimum_probability=0.0) for item in corpus]

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        similarity_matrix(corpus, dictionary)

        # visualise_LDA(lda_model, corpus, dictionary)

        # outputs results in a txt file
        write_result(lda_model, avg_topic_coherence, topic_dist)

        # Clustering by Kmeans
        num_topics = 8  # need to write a function to find out the optimal numb of topics
        clusters_result = clusterByKMeans(num_topics=num_topics, lda_model=lda_model, y_true=msg_type, docs=docs)

        # Getting the Performance matrix (accuracy, confusion matrix, precision)
        visualise_confusion(clusters_result=clusters_result)

    def clusterByKMeans(num_topics, lda_model, y_true, docs):
        """ Clusters the output of LDA by Kmeans
        Returns y true and y predicted as well. And Kmeans visualisation."""
        corpus = create_corpus(docs)
        topic_dist = [lda_model.get_document_topics(item, minimum_probability=0.0) for item in corpus]
        X = pd.DataFrame(topic_dist)  # Dataframe of the result. Use Jupyter notebook to view.

        entry_num = 1  # index one -> the probability of message in topics

        # Removing the id from the tuple, leaving the probablity of each word being in topic
        for row in X.iterrows():
            for i in range(0, num_topics):
                row[entry_num][i] = row[entry_num][i][1]

        # PCA first to reduce dimensionality for visualisation
        pca = PCA(n_components=2)
        PC = pca.fit_transform(X)

        # Applying Kmeans to get labels(cluster no)
        kmeans = KMeans(n_clusters=num_topics, n_init=10).fit_predict(PC)

        # Dataframe with labels
        Y = pd.DataFrame()
        Y["true_labels"] = y_true
        cluster_predicted = kmeans.tolist()
        Y["pred_labels"] = cluster_predicted
        Y.groupby("pred_labels")
        label_dict = {}
        for (i, row) in Y.iterrows():
            if row["pred_labels"] in label_dict:
                label_dict[row["pred_labels"]].append(row["true_labels"])
            else:
                label_dict[row["pred_labels"]] = [row["true_labels"]]

        y_pred = []
        for i in label_dict:
            # Labelling the predicted cluster
            maj = majority_element(label_dict[i])
            cluster_maj = np.full((1, len(label_dict[i])), maj).tolist()[0]
            # print(cluster_predicted)
            y_pred.extend(cluster_maj)  # Adding to the list of predicted labels for cluster

        # Plotting Kmeans
        # iterating through no of categories
        print(np.unique(kmeans))
        for i in np.unique(kmeans):
            plotx = []
            ploty = []
            for j in range(PC.shape[0]):
                if kmeans[j] == i:
                    plotx.append(PC[j][0])
                    ploty.append(PC[j][1])

            # Plotting the graph
            plt.scatter(plotx, ploty, label=i)  # projected points to the axis

        return (y_pred, y_true, np.unique(kmeans))

    def tf_idf(corpus):
        """Using TF/IDF to vectorize the data
        Returns tfidf weighted corpus"""
        tfidf = TfidfModel(corpus)  # fit model
        # tfidf = [model[corpus[i]] for i in range(len(corpus))]
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    def identify_msg_type(message):
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

    def visualise_LDA(lda_model, corpus, dictionary):
        """Visualise the LDA results"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(visualisation, 'LDA_Visualisation.html')

    def write_result(lda_model, avg_topic_coherence, topic_dist):
        """Create a text document of the result"""
        with open("result4.txt", "w") as f:
            # pprint(topic_dist, stream=f)
            print(topic_dist, file=f)
            print('Average topic coherence: %.4f.' % avg_topic_coherence, file=f)

    def load_lda():
        """Loading the LDA model from save file
        Returns the LDA Model"""
        corpus = create_corpus()
        temp_file = datapath("model")
        lda_model = gensim.models.LdaModel.load(temp_file)

        return lda_model

    def import_message(file_path, importLayer):
        """Abstracted function to import messages"""
        message = PCAPImporter.readFile(file_path, importLayer=importLayer).values()
        return message

    def write_message(message):
        """Outputs the message in data.txt
        For LDA to process """
        datafile = open("data.txt", "w")
        symbol = Symbol(messages=message)
        # For Symbol
        datafile.writelines(str(symbol) + "\n")

        datafile.close()

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

    def parse_input(text):
        return text.strip("\n").strip(" ").strip("b")

    def tokenize_hex(text):
        # re.split(r'\\x'+'\\',text)
        return text.split("\\")

    def is_hex(text):
        return text != "\'"

    def parse_hex(text):
        return text.strip("x")

    def msg_to_bytes():
        """ Breaking Messages into bytes
        Returns a list of Messages
        """
        f = open("data.txt", "r")
        # print(sent_tokenize(text))
        text = f.readlines()
        doc = []
        for line in text:
            parsed_hex = []
            if "\\x" in line:
                line = parse_input(line)
                tokenized_hex = tokenize_hex(line)
                for hex in tokenized_hex:
                    if is_hex(hex):
                        parsed_hex.append(parse_hex(hex))
                doc.append(parsed_hex)
        return doc

    def n_gram_message(docs):
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

    def filter_tokens(dictionary):
        """ Filter out words that occur less than "no_below" documents, or more than "no_above" of the documents.
        Returns dictionary with filtered tokens"""
        no_below = 10
        no_above = 0.2
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)

        return dictionary

    def create_dict(docs):
        """ Create a dictionary representation of the documents."""
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)
        return dictionary

    def create_corpus(docs):
        """Returns a TF/IDF Weighted corpus"""
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)
        # Create a dictionary representation of the documents.
        # Bag-of-words representation of the documents.
        corpus = [dictionary.doc2bow(doc) for doc in docs]  # output (ID:frequency)
        # Using Tf-Idf
        corpus_tfidf = tf_idf(corpus)  # Gensim object
        return corpus_tfidf
