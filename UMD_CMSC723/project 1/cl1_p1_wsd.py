"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""

"""
 read one of train, dev, test subsets 
 
 subset - one of train, dev, test
 
 output is a tuple of three lists
 	labels: one of the 6 possible senses <cord, division, formation, phone, product, text >
 	targets: the index within the text of the token to be disambiguated
 	texts: a list of tokenized and normalized text input (note that there can be multiple sentences)

"""
import nltk
from nltk.corpus import stopwords
import numpy as np
import collections
import itertools
from sklearn.preprocessing import LabelEncoder


def read_dataset(subset):
    labels = []
    texts = []
    targets = []
    if subset in ['train', 'dev', 'test']:
        with open('wsd_line_dataset/wsd_' + subset + '.txt') as inp_hndl:
            for example in inp_hndl:
                label, text = example.strip().split('\t')
                text = nltk.word_tokenize(text.lower().replace('" ', '"'))
                if 'line' in text:
                    ambig_ix = text.index('line')
                elif 'lines' in text:
                    ambig_ix = text.index('lines')
                else:
                    ldjal
                targets.append(ambig_ix)
                labels.append(label)
                texts.append(text)
        return labels, targets, texts
    else:
        print('>>>> invalid input !!! <<<<<')


"""
computes f1-score of the classification accuracy

gold_labels - is a list of the gold labels
predicted_labels - is a list of the predicted labels

output is a tuple of the micro averaged score and the macro averaged score

"""
import sklearn.metrics


def eval(gold_labels, predicted_labels):
    return (sklearn.metrics.f1_score(gold_labels, predicted_labels, average='micro'),
            sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro'))


"""
a helper method that takes a list of predictions and writes them to a file (1 prediction per line)
predictions - list of predictions (strings)
file_name - name of the output file
"""


def write_predictions(predictions, file_name):
    with open(file_name, 'w') as outh:
        for p in predictions:
            outh.write(p + '\n')


"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""


def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
                                  dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels):
    """
    **Your final classifier implementation of part 2 goes here**
    """
    ## Trains a naive bayes model with bag of words features
    words = list(set(list(itertools.chain(*train_texts))))
    sum_words = sum([len(sentence) for sentence in train_texts])
    labels = list(set(train_labels))
    p_s = np.array([collections.Counter(train_labels)[s] / len(train_labels) for s in labels])
    matrix_count = np.zeros((len(words), len(labels)))
    for index in range(len(train_texts)):
        label_ix = labels.index(train_labels[index])
        for word in train_texts[index]:
            word_ix = words.index(word)
            matrix_count[word_ix, label_ix] += 1
    # add-1 smoothing
    matrix_count += 1
    # calculate sum of counts for each label
    sum_s = np.array([np.sum(matrix_count[:, i]) for i in range(len(labels))])
    # log probability of word-document matrix
    matrix_ws = np.log(matrix_count / sum_s)

    ## Compute accuracy on the test set
    nb_test_pred = []
    for i in range(len(test_texts)):
        total_log_p = np.zeros((1, 6))
        for j in test_texts[i]:
            if j in words:
                total_log_p += matrix_ws[words.index(j)]
        y_hat = labels[np.argmax(total_log_p)]
        nb_test_pred.append(y_hat)
    # return accuracy
    return eval(test_labels, nb_test_pred)


"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""


def run_bow_perceptron_classifier(train_texts, train_targets, train_labels,
                                  dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels):
    bag = set()
    for sentence in train_texts:
        for word in sentence:
            bag.add(word)
    bag = list(bag)

    # bag of word feature
    numofSentence = len(train_labels)
    numofFeature = len(bag)
    bagfeature = np.zeros(shape=(numofSentence, (numofFeature + 1)))

    c = 0
    for sentence in train_texts:
        for word in sentence:
            j = bag.index(word)
            # print(i)
            bagfeature[c, j] += 1
        c += 1

    bagfeature[:, -1] = 1
    # perceptron learning
    weights = dict()
    weights['phone'] = np.zeros(shape=(1, (numofFeature + 1)))
    weights['text'] = np.zeros(shape=(1, (numofFeature + 1)))
    weights['cord'] = np.zeros(shape=(1, (numofFeature + 1)))
    weights['division'] = np.zeros(shape=(1, (numofFeature + 1)))
    weights['formation'] = np.zeros(shape=(1, (numofFeature + 1)))
    weights['product'] = np.zeros(shape=(1, (numofFeature + 1)))

    iterations = 20
    # output=[]
    for iteration in range(iterations):
        for i in range(numofSentence):
            trainresult = dict()
            feature = bagfeature[i, :]
            for d in weights:
                trainresult[d] = np.dot(weights[d], feature)[0]

                # find argmax
            actuallabel = train_labels[i]
            predictlabel = list(trainresult.keys())[0]
            for key in trainresult:
                if trainresult[key] > trainresult[predictlabel]:
                    predictlabel = key

            if predictlabel != actuallabel:
                weights[predictlabel] -= feature
                weights[actuallabel] += feature
        '''
        #for Part 3.2
        testontraining=[]
        for i in range(len(train_labels)):
            result=dict()
            testfeature=bagfeature[i,:]
            for d in weights:
                result[d]=np.dot(weights[d],testfeature)[0]

            maxresult=max(result, key=result.get)
            testontraining.append(maxresult)
        output.append(eval(train_labels,testontraining))
    print(output)
    print(len(output))
        '''

    '''
            #for Part 3.1
            a=weights['phone']
            print(a)

            outputofA=''
            outputofB=''
            b=weights['text']
            print(b)

            for i in range(numofFeature):
                if a.item(i) !=0:
                    outputofA+=str(bag[i])+'_'+ 'phone'+ ':'+str(a.item(i))+','


            for i in range(numofFeature):
                if b.item(i) !=0:
                    outputofB+=str(bag[i]) +'_'+ 'text'+':'+str(b.item(i))+','
            print(outputofA)
            print(outputofB)

            break
        break
    '''

    # bag of word test feature
    testbagfeature = np.zeros(shape=(len(test_labels), (numofFeature + 1)))

    a = 0
    for sentence in test_texts:
        for word in sentence:
            if word in bag:
                b = bag.index(word)
                testbagfeature[a, b] += 1
        a += 1

    testbagfeature[:, -1] = 1

    # prediction for testing set
    test = []
    for i in range(len(test_labels)):
        result = dict()
        testfeature = testbagfeature[i, :]
        for d in weights:
            result[d] = np.dot(weights[d], testfeature)[0]

        maxresult = max(result, key=result.get)
        test.append(maxresult)

    write_predictions(test, 'q3p3.txt')
    test_scores = eval(test_labels, test)
    return test_scores


"""
Trains a naive bayes model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""


def run_extended_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
                                           dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels):
    """
    **Your final implementation of Part 4 with perceptron classifier**
    """
    ## basic classifier
    words = list(set(list(itertools.chain(*train_texts))))
    sum_words = sum([len(sentence) for sentence in train_texts])
    labels = list(set(train_labels))
    p_s = np.array([collections.Counter(train_labels)[s] / len(train_labels) for s in labels])
    matrix_count = np.zeros((len(words), len(labels)))
    for index in range(len(train_texts)):
        label_ix = labels.index(train_labels[index])
        for word in train_texts[index]:
            word_ix = words.index(word)
            matrix_count[word_ix, label_ix] += 1
    # add-1 smoothing
    matrix_count += 1
    # calculate sum of counts for each label
    sum_s = np.array([np.sum(matrix_count[:, i]) for i in range(len(labels))])
    # log probability of word-document matrix
    matrix_ws = np.log(matrix_count / sum_s)

    ## feature 1 parameters
    positions = np.array([train_targets[i] / len(train_texts[i]) for i in range(len(train_texts))])
    positions_range = np.arange(0.2, 1.2, 0.2)
    matrix_pos_count = np.zeros((5, 6))
    for i in range(len(positions)):
        label_ix = labels.index(train_labels[i])
        for j in range(5):
            if positions[i] < positions_range[j]:
                matrix_pos_count[j, label_ix] += 1
                break
    matrix_pos_p = np.log(matrix_pos_count / sum(matrix_pos_count))

    ## feature 2 parameters
    stop = set(stopwords.words('english'))
    stopwords_ratio_n = np.array([len([i for i in train_texts[j] if i in stop])
                                  / len(train_texts[j])
                                  for j in range(len(train_texts))])
    stopwords_ratio_range = np.arange(0.2, 1.2, 0.2)
    matrix_stopwords_ratio_count = np.zeros((5, 6))
    for i in range(len(stopwords_ratio_n)):
        label_ix = labels.index(train_labels[i])
        for j in range(5):
            if stopwords_ratio_n[i] < stopwords_ratio_range[j]:
                matrix_stopwords_ratio_count[j, label_ix] += 1
                break
    # add-1 smoothing
    matrix_stopwords_ratio_count += 1
    matrix_stopwords_ratio_p = np.log(matrix_stopwords_ratio_count / sum(matrix_stopwords_ratio_count))

    ## Compute accuracy on the test set
    nb_test_pred = []
    for i in range(len(test_texts)):
        total_log_p = np.zeros((1, 6))
        for j in test_texts[i]:
            if j in words:
                total_log_p += matrix_ws[words.index(j)]
        # add feature 1
        pos_ix = int(test_targets[i] / len(test_texts[i]) // 0.2)
        total_log_p += matrix_pos_p[pos_ix]
        # add feature 2
        stopwords_ix = int(len([k for k in test_texts[i] if k in stop]) / len(test_texts[i])
                           // 0.2)
        total_log_p += matrix_stopwords_ratio_p[stopwords_ix]
        y_hat = labels[np.argmax(total_log_p)]
        nb_test_pred.append(y_hat)
    # return accuracy
    return eval(test_labels, nb_test_pred)


"""
Trains a perceptron model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""


def run_extended_bow_perceptron_classifier(train_texts, train_targets, train_labels,
                                           dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels):
    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))
    bag = set()
    for sentence in train_texts:
        for word in sentence:
            bag.add(word)
    bag = list(bag)

    # bag of word feature
    numofSentence = len(train_labels)
    numofFeature = len(bag)
    bagfeature = np.zeros(shape=(numofSentence, (numofFeature + 3)))

    c = 0
    for sentence in train_texts:
        for word in sentence:
            j = bag.index(word)
            # print(i)
            bagfeature[c, j] += 1
        c += 1

    bagfeature[:, -3] = 1

    # add addtional first feature for target position for training set
    for i in range(len(train_targets)):
        targetPosition = round(train_targets[i] / len(train_texts[i]), 1)
        if targetPosition > 0.8:
            bagfeature[i, -2] = 5
        elif targetPosition > 0.6:
            bagfeature[i, -2] = 4
        elif targetPosition > 0.4:
            bagfeature[i, -2] = 3
        elif targetPosition > 0.2:
            bagfeature[i, -2] = 2
        elif targetPosition > 0:
            bagfeature[i, -2] = 1

    # add additional second feature for the ratio between lenth of stopwords and lenth of non-stopwords

    location = 0
    for sentence in train_texts:
        numberofword = 0
        numberofstopWord = 0
        for word in sentence:
            if word in stopWords:
                numberofstopWord += 1
            else:
                numberofword += 1
        ratio = round(numberofstopWord / numberofword, 2)
        if ratio > 0.8:
            bagfeature[location, -1] = 5
        elif ratio > 0.6:
            bagfeature[location, -1] = 4
        elif ratio > 0.4:
            bagfeature[location, -1] = 3
        elif ratio > 0.2:
            bagfeature[location, -1] = 2
        elif ratio > 0:
            bagfeature[location, -1] = 1
        location += 1

    # perceptron learning
    weights = dict()
    weights['phone'] = np.zeros(shape=(1, (numofFeature + 3)))
    weights['text'] = np.zeros(shape=(1, (numofFeature + 3)))
    weights['cord'] = np.zeros(shape=(1, (numofFeature + 3)))
    weights['division'] = np.zeros(shape=(1, (numofFeature + 3)))
    weights['formation'] = np.zeros(shape=(1, (numofFeature + 3)))
    weights['product'] = np.zeros(shape=(1, (numofFeature + 3)))

    iterations = 20
    for iteration in range(iterations):
        for i in range(numofSentence):
            trainresult = dict()
            feature = bagfeature[i, :]
            for d in weights:
                trainresult[d] = np.dot(weights[d], feature)[0]

                # find argmax
            actuallabel = train_labels[i]
            predictlabel = list(trainresult.keys())[0]
            for key in trainresult:
                if trainresult[key] > trainresult[predictlabel]:
                    predictlabel = key

            if predictlabel != actuallabel:
                weights[predictlabel] -= feature
                weights[actuallabel] += feature

                # bag of word test feature
    testbagfeature = np.zeros(shape=(len(test_labels), (numofFeature + 3)))

    a = 0
    for sentence in test_texts:
        for word in sentence:
            if word in bag:
                b = bag.index(word)
                testbagfeature[a, b] += 1
        a += 1

    testbagfeature[:, -3] = 1

    # add addtional feature for target position for testing set
    for i in range(len(test_targets)):
        targetPosition = round(test_targets[i] / len(test_texts[i]), 1)
        if targetPosition > 0.8:
            testbagfeature[i, -2] = 5
        elif targetPosition > 0.6:
            testbagfeature[i, -2] = 4
        elif targetPosition > 0.4:
            testbagfeature[i, -2] = 3
        elif targetPosition > 0.2:
            testbagfeature[i, -2] = 2
        elif targetPosition > 0:
            testbagfeature[i, -2] = 1

    # add addtional feature for ratio for testing set
    location = 0
    for sentence in test_texts:
        numberofword = 0
        numberofstopWord = 0
        for word in sentence:
            if word in stopWords:
                numberofstopWord += 1
            else:
                numberofword += 1
        ratio = round(numberofstopWord / numberofword, 2)
        if ratio > 0.8:
            testbagfeature[location, -1] = 5
        elif ratio > 0.6:
            testbagfeature[location, -1] = 4
        elif ratio > 0.4:
            testbagfeature[location, -1] = 3
        elif ratio > 0.2:
            testbagfeature[location, -1] = 2
        elif ratio > 0:
            testbagfeature[location, -1] = 1
        location += 1

        # prediction for testing set
    test = []
    for i in range(len(test_labels)):
        result = dict()
        testfeature = testbagfeature[i, :]
        for d in weights:
            result[d] = np.dot(weights[d], testfeature)[0]

        maxresult = max(result, key=result.get)
        test.append(maxresult)

    write_predictions(test, 'q4p4_pn.txt')
    test_scores = eval(test_labels, test)
    return (test_scores)


if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    # running the classifier
    # test_scores = run_extended_bow_naivebayes_classifier(train_texts, train_targets, train_labels, dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)
    # print(test_scores)
    clfs = [run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels),
            run_bow_perceptron_classifier(train_texts, train_targets, train_labels, dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels),
            run_extended_bow_naivebayes_classifier(train_texts, train_targets, train_labels, dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels),
            run_extended_bow_perceptron_classifier(train_texts, train_targets, train_labels, dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)]
    for i in clfs:
        print(i)
