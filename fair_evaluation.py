import sys
import numpy as np
import pandas as pd
from metrics import strict, loose_macro, loose_micro

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
    if len(sys.argv) != 3:
        print("Usage: Python fair_evalaution.py dataset log_file")
        sys.exit(0)
    else:
        dataset = sys.argv[1]
        if dataset == 'BBN':
            num_of_labels = 47
        elif dataset == 'Wiki':
            num_of_labels = 128
        elif dataset == 'OntoNotes':
            num_of_labels = 89

        log_file = sys.argv[2]

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

        raw_target_no, raw_target_label = generate_no_and_label(open('./Intermediate/' + dataset + '/mention_type_test.txt', 'r'), num_of_labels)
        
        dev_target_no, dev_target_label, test_target_no, test_target_label = filter_mentions(raw_target_no, raw_target_label, dev_set, test_set)

        raw_prediction_no, raw_prediction_label = generate_no_and_label(open('./Results/' + dataset + '/mention_type_pl_warp_bipartite.txt', 'r'), num_of_labels)

        dev_prediction_no, dev_prediction_label, test_prediction_no, test_prediction_label = filter_mentions(raw_prediction_no, raw_prediction_label, dev_set, test_set)

        print('size of dev and train set')
        print(len(dev_target_no), len(test_target_no))
        # write result in same format as used by our other models
        data = pd.DataFrame(columns=('train_cost',
            'dev_cost',
            'test_cost',
            'dev_acc',
            'dev_ma_F1',
            'dev_mi_F1',
            'test_acc',
            'test_ma_F1',
            'test_mi_F1'))

        print('Dev set result')
        current_result = []
        current_result.append(0)
        current_result.append(0)
        current_result.append(0)
        current_result.append(strict(dev_prediction_label, dev_target_label))
        current_result.append(loose_macro(dev_prediction_label, dev_target_label))
        current_result.append(loose_micro(dev_prediction_label, dev_target_label))
        current_result.append(strict(test_prediction_label, test_target_label))
        current_result.append(loose_macro(test_prediction_label, test_target_label))
        current_result.append(loose_micro(test_prediction_label, test_target_label))
        data.loc[len(data)] = current_result
        data.to_csv(log_file, index=False)
