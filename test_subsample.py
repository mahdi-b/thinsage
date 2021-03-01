import numpy as np
import pytest
from dummy_object import DummyObj
from thinsage_subsample import random_subsample
from thinsage_subsample import stratified_subsample_balanced

def test_random_subsample():
    data_size = 100
    test_list = np.arange(0, data_size)
    test_size = 20
    test_sub_sample = random_subsample(test_list, test_size)
    assert len(test_sub_sample) == test_size
    assert type(test_list[0]) is type(test_sub_sample[0])
    for i in range(test_size):
        assert test_sub_sample[i] in test_list

    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, data_size + 1)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)

def test_random_subsample_obj():
    data_size = 100
    test_size = 20
    # classes = ['red', 'blue', 'green']
    test_list = []
    # class_list = []
    for i in range(data_size):
        test_list.append(DummyObj())
        # class_list.append(classes[i%3])

    test_sub_sample = random_subsample(test_list, test_size)
    assert len(test_sub_sample) == test_size
    assert type(test_list[0]) is type(test_sub_sample[0])
    UUID_list = []
    for i in range(test_size):
        assert test_sub_sample[i] in test_list
        if len(UUID_list) > 0:
            assert test_sub_sample[i].UUID not in UUID_list
        UUID_list.append(test_sub_sample[i].UUID)


    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, data_size + 1)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)

def test_stratified_balanced_subsample():
    data_size = 100
    test_list = np.arange(0,data_size)
    num_classes = 6
    test_size = 20
    test_class_list = [i % num_classes for i in range(data_size)]
    test_sub_sample, sub_class_list = stratified_subsample_balanced(test_list, test_class_list, test_size)
    assert len(test_sub_sample) == test_size
    assert type(test_list[0]) is type(test_sub_sample[0])
    for i in range(test_size):
        assert test_sub_sample[i] in test_list
        # check returned class is same as the original
        assert sub_class_list[i] == (test_sub_sample[i] % num_classes)

    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, data_size + 1)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 0)
    with pytest.raises(Exception) as e_info:
        bad_sub_sample = random_subsample(test_list, 1.3)
    print(test_sub_sample, sub_class_list)