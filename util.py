import numpy as np


class Predict_data:
    def __init__(
        self, test_data, test_ids, filtered_data, filtered_labels, filtered_ids
    ):
        self.test_data = test_data
        self.test_ids = test_ids
        self.filtered_data = filtered_data
        self.filtered_labels = filtered_labels
        self.filtered_ids = filtered_ids


def extract_sample(data, labels, ids, classify_id):
    sample_index = None
    for i in xrange(len(ids)):
        if ids[i] == classify_id:
            sample_index = i
    if not sample_index:
        print("Unable to find: " + str(classify_id) + " in dataset")
        exit
    start_len = len(data)
    test_data = np.array(data[sample_index]).reshape(1, -1)
    filtered_data = np.delete(data, sample_index, axis=0)
    filtered_labels = np.delete(labels, sample_index, axis=0)
    filtered_ids = np.delete(ids, sample_index, axis=0)
    test_ids = np.array([classify_id]).reshape(1, -1)
    end_len = len(filtered_data)
    assert start_len - end_len == len(test_data)
    return Predict_data(
        test_data, test_ids, filtered_data, filtered_labels, filtered_ids
    )
