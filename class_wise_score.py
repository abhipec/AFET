import sys
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import precision_recall_fscore_support
#pylint: disable=too-many-locals
def class_wise_f1_score(prediction,
                        target,
                        num_to_label,
                        sort_id=3,
                        filename=None):
    """
    Compute class wise f1 score.

    Args:
        prediction: A numpy array of shape [batch_size, number_of_labels].
        target: A numpy array of shape [batch_size, number_of_labels].
        num_to_label: A dictionary with index to label name mapping.
        sord_id: Index on which sorting is done. [P, R, F1, S].
        filename: If filename is specified, write csv with sorted output.
    """
    assert prediction.shape == target.shape, \
        "Prediction and target shape should match."
    assert sort_id < 4 and sort_id > -1, \
        "Sort id should be from 0,1,2,3"
    _, number_of_labels = prediction.shape
    output = {}
    for i in range(number_of_labels):
        precision, recall, f_score, _ = precision_recall_fscore_support(target[:, i],
                                                                        prediction[:, i],
                                                                        average='binary')
        output[num_to_label[i]] = [precision, recall, f_score, np.sum(target[:, i]), i]
    # sorting based on score
    sorted_out = sorted(output.items(), key=lambda e: e[1][sort_id], reverse=True)
    if filename:
        # dump score in a file
        with open(filename, 'w') as file_p:
            writer = csv.writer(file_p)
            for key, value in sorted_out:
                writer.writerow([key] + value)
    return output

def generate_remove_list(f):
    remove = []
    for row in f:
        part1, part2 = row.split('\t')
        if '-1' in part1:
            remove.append(int(part2[:-1]))

    #print(len(remove))
    f.close()
    return set(remove)

def generate_num_to_label(filename):
    """
    Generate number to label"".
    """
    num_to_label = {}
    with open(filename) as file_p:
        for row in filter(None, file_p.read().split('\n')):
            part1, part2, _ = row.split('\t')
            num_to_label[int(part2)] = part1
    return num_to_label

def generate_no_and_label(f, num_of_labels):
    no = []
    labels = []
    store = {}
    for row in f:
        part1, part2, part3 = row.split('\t')
        if int(part1) not in store:
            store[int(part1)] = np.zeros(num_of_labels, dtype=np.float32)
        store[int(part1)][int(part2)] = 1
    for key in sorted(store):
        no.append(key)
        labels.append(store[key])
    f.close()
    return np.array(no, dtype=np.uint32), np.array(labels, dtype=np.float32)


def filter_mentions(no, labels, dev, test):
    dev_nos = []
    dev_labels = []
    test_nos = []
    test_labels = []
    for i in range(len(no)):
        if no[i] in test:
            test_nos.append(no[i])
            test_labels.append(labels[i])
        elif no[i] in dev:
            dev_nos.append(no[i])
            dev_labels.append(labels[i])
    return np.array(dev_nos, dtype=np.uint32), np.array(dev_labels, dtype=np.float32), np.array(test_nos, dtype=np.uint32), np.array(test_labels, dtype=np.float32)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: Python fair_evalaution.py dataset")
        sys.exit(0)
    else:
        dataset = sys.argv[1]
        if dataset == 'BBN':
            num_of_labels = 47
        elif dataset == 'Wiki':
            num_of_labels = 128
        elif dataset == 'OntoNotes':
            num_of_labels = 89

        with open('./sanitized/' + dataset + '/sanitized_mention_dev.txt') as file_p:
            dev_ids = file_p.read().split('\n')
        with open('./sanitized/' + dataset + '/sanitized_mention_test.txt') as file_p:
            test_ids = file_p.read().split('\n')

        with open('./Intermediate/' + dataset + '/mention.txt') as file_p:
            uid_to_number = {}
            number_to_uid = {}
            rows = list(filter(None, file_p.read().split('\n')))
            for row in rows:
                uid, number = row.split('\t')
                uid_to_number[uid] = int(number)
                number_to_uid[int(number)] = uid

        dev_set = set()
        for uid in dev_ids:
            dev_set.add(uid_to_number[uid])

        test_set = set()
        for uid in test_ids:
            test_set.add(uid_to_number[uid])
        num_to_label = generate_num_to_label('./Intermediate/' + dataset + '/type.txt')
        raw_target_no, raw_target_label = generate_no_and_label(open('./Intermediate/' + dataset + '/mention_type_test.txt', 'r'), num_of_labels)
        
        dev_target_no, dev_target_label, test_target_no, test_target_label = filter_mentions(raw_target_no, raw_target_label, dev_set, test_set)

        raw_prediction_no, raw_prediction_label = generate_no_and_label(open('./Results/' + dataset + '/mention_type_pl_warp_bipartite.txt', 'r'), num_of_labels)

        dev_prediction_no, dev_prediction_label, test_prediction_no, test_prediction_label = filter_mentions(raw_prediction_no, raw_prediction_label, dev_set, test_set)

        class_wise_f1_score(test_prediction_label,
                            test_target_label,
                            num_to_label,
                            filename='class_wise_score_' + dataset + '.csv')

        # dump results 
        with open('result_mentions.txt', 'w') as file_p:
            for number in test_prediction_no:
                file_p.write(number_to_uid[number] + '\n')
