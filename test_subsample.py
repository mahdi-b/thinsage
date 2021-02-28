import numpy as np
import pytest
from thinsage_subsample import random_subsample
from thinsage_subsample import stratified_subsample_balanced

def test_random_subsample():
    test_list = np.arange(0, 10)
    test_size = 3
    test_sub_sample = random_subsample(test_list, test_size)
    assert len(test_sub_sample) == test_size
    assert type(test_list[0]) is type(test_sub_sample[0])
    for i in range(test_size):
        assert test_sub_sample[i] in test_list

    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 11)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)

def test_stratified_balanced_subsample():
    test_list = np.arange(0,10)
    num_classes = 2
    test_size = 5
    test_class_list = [i % num_classes for i in range(10)]
    test_sub_sample, sub_class_list = stratified_subsample_balanced(test_list, test_class_list, test_size)
    assert len(test_sub_sample) == test_size
    assert type(test_list[0]) is type(test_sub_sample[0])
    for i in range(test_size):
        assert test_sub_sample[i] in test_list
        # check returned class is same as the original
        assert sub_class_list[i] == (test_sub_sample[i] % num_classes)

    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 11)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)

