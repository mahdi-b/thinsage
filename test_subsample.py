import numpy as np
import pytest
from dummy_object import DummyObj
from thinsage_subsample import random_subsample
from thinsage_subsample import stratified_subsample_balanced


def test_random_subsample():
    # initialize data
    data_size = 100
    test_list = np.arange(0, data_size)
    test_size = 20
    test_sub_sample = random_subsample(test_list, test_size)
    # check size of returned subsamples is as desired
    assert len(test_sub_sample) == test_size
    # check type of data is unchanged
    if type(test_list[0]) is list:
        assert type(test_list[0][0]) is type(test_sub_sample[0][0])
    else:
        assert type(test_list[0]) is type(test_sub_sample[0])

    for i in range(test_size):
        # check subsamples are from the original data
        assert test_sub_sample[i] in test_list
        # check no duplicates
        for j in range(i+1, test_size):
            assert test_sub_sample[i] != test_sub_sample[j]

    # check invalid size parameter raises exception
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, data_size + 1)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)


def test_random_subsample_obj():
    # initialize data
    data_size = 100
    test_size = 20
    test_list = [DummyObj() for i in range(data_size)]

    # subsample data
    test_sub_sample = random_subsample(test_list, test_size)
    # check number of subsamples is as desired
    assert len(test_sub_sample) == test_size
    # check type of subsamples is unchanged
    if type(test_list[0]) is list:
        assert type(test_list[0][0]) is type(test_sub_sample[0][0])
    else:
        assert type(test_list[0]) is type(test_sub_sample[0])

    # UUID_list = []
    for i in range(test_size):
        # check returned samples are from original data
        assert test_sub_sample[i] in test_list
        # check no duplicates
        for j in range(i+1, test_size):
            assert test_sub_sample[i].UUID != test_sub_sample[j].UUID
        # if len(UUID_list) > 0:
        #     assert test_sub_sample[i].UUID not in UUID_list
        # UUID_list.append(test_sub_sample[i].UUID)


def test_stratified_balanced_subsample():
    # initialize data and class list
    data_size = 100
    test_list = np.arange(0,data_size)
    num_classes = 6
    test_size = 20
    test_class_list = [i % num_classes for i in range(data_size)]
    # subsample data
    test_sub_sample, sub_class_list = stratified_subsample_balanced(test_list, test_class_list, test_size)
    # check returned number of samples equals desired size
    assert len(test_sub_sample) == test_size
    # check type of data remained the same
    if type(test_list[0]) is list:
        assert type(test_list[0][0]) is type(test_sub_sample[0][0])
    else:
        assert type(test_list[0]) is type(test_sub_sample[0])
    for i in range(test_size):
        # check returned samples are from original data
        assert test_sub_sample[i] in test_list
        # check returned class is same as the original
        assert sub_class_list[i] == (test_sub_sample[i] % num_classes)
        # check no duplicates are returned
        for j in range(i+1, test_size):
            assert test_sub_sample[i] != test_sub_sample[j]

    # check invalid size parameters raise exceptions
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, data_size + 1)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)
    # print(test_sub_sample, sub_class_list)

    # check first that number of uncique classes in returned data are same as
    # original number of classes
    assert len(np.unique(sub_class_list)) == num_classes
    # check proportions of data from each class are close to balanced
    # (within) err of +/- 1
    lengths = {}
    for i in range(len(sub_class_list)):
        if sub_class_list[i] not in lengths.keys():
            lengths[sub_class_list[i]] = 1
        else:
            lengths[sub_class_list[i]] += 1
    for key1 in lengths.keys():
        for key2 in lengths.keys():
            if key1 == key2:
                continue
            assert np.abs(lengths[key1] - lengths[key2]) <= 1


def test_stratified_subsample_balanced_obj():
    # initialize data
    data_size = 100
    classes = ['red', 'blue', 'green']
    num_classes = len(classes)
    test_list = []
    class_list = []
    for i in range(data_size):
        this_class = np.random.choice(classes)
        test_list.append(DummyObj(class_id=this_class))
        class_list.append(this_class)
    # sub sample from data
    test_size = 45
    test_sub_sample, sub_class_list = stratified_subsample_balanced(test_list, class_list, test_size)
    # check returned samples is equal to specified size
    assert len(test_sub_sample) == test_size
    # check that type of items returned is same as when it went in
    if type(test_list[0]) is list:
        assert type(test_list[0][0]) is type(test_sub_sample[0][0])
    else:
        assert type(test_list[0]) is type(test_sub_sample[0])

    for i in range(test_size):
        # check all that are returned are from the original set
        assert test_sub_sample[i] in test_list
        # check returned class is same as the original
        assert test_sub_sample[i].class_id == sub_class_list[i]
        # check no duplicates were returned
        for j in range(i+1, test_size):
            assert test_sub_sample[i].UUID != test_sub_sample[j].UUID

    # check first that number of unique classes in returned data are same as
    # original number of classes
    assert len(np.unique(sub_class_list)) == num_classes
    # check proportions of data from each class are close to balanced
    # (within) err of +/- 1
    lengths = {}
    for i in range(len(sub_class_list)):
        if sub_class_list[i] not in lengths.keys():
            lengths[sub_class_list[i]] = 1
        else:
            lengths[sub_class_list[i]] += 1
    for key1 in lengths.keys():
        for key2 in lengths.keys():
            if key1 == key2:
                continue
            assert np.abs(lengths[key1] - lengths[key2]) <= 1


def test_stratified_subsample_imbalanced_obj():
    # initialize data
    data_size = 100
    classes = ['red', 'blue', 'green', 'yellow']
    num_classes = len(classes)
    test_list = []
    class_list = []
    proportions = {}
    for this_class in classes:
        proportions[this_class] = 0
    for i in range(data_size):
        this_class = np.random.choice(classes)
        test_list.append(DummyObj(class_id=this_class))
        class_list.append(this_class)
        proportions[this_class] += 1

    for this_class in classes:
        proportions[this_class] = float(proportions[this_class] / data_size)

    # sub sample from data
    test_size = 25
    test_sub_sample, sub_class_list = stratified_subsample_balanced(test_list, class_list, test_size)
    # check returned samples is equal to specified size
    assert len(test_sub_sample) == test_size
    # check that type of items returned is same as when it went in
    if type(test_list[0]) is list:
        assert type(test_list[0][0]) is type(test_sub_sample[0][0])
    else:
        assert type(test_list[0]) is type(test_sub_sample[0])

    for i in range(test_size):
        # check all that are returned are from the original set
        assert test_sub_sample[i] in test_list
        # check returned class is same as the original
        assert test_sub_sample[i].class_id == sub_class_list[i]
        # check no duplicates were returned
        for j in range(i+1, test_size):
            assert test_sub_sample[i].UUID != test_sub_sample[j].UUID

    # check first that number of unique classes in returned data are same as
    # original number of classes
    assert len(np.unique(sub_class_list)) == num_classes
    # check proportions of data from each class are close to original proportions
    # (within) err of +/- num_classes
    # get number of items per class
    lengths = {}
    for i in range(len(sub_class_list)):
        if sub_class_list[i] not in lengths.keys():
            lengths[sub_class_list[i]] = 1
        else:
            lengths[sub_class_list[i]] += 1
    # check against original proportions
    for key in lengths.keys():
        diff = np.abs(lengths[key] - (proportions[key] * test_size))
        # best accuracy achievable?
        assert diff <= num_classes
