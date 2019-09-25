import numpy as np 
import h5py as h5

def load_dataset():
    train_dataset = h5.File("dataset/train_set.hdf5",'r')
    train_set_input = np.array(train_dataset["images_train"][:])
    train_set_label = np.array(train_dataset["labels_train"][:]).astype("uint8")

    test_dataset = h5.File("dataset/dev_set.hdf5",'r')
    test_set_input = np.array(test_dataset["images_dev"][:]) 
    test_set_label = np.array(test_dataset["labels_dev"][:]).astype("uint8")
    
    return train_set_input,train_set_label,test_set_input,test_set_label

def convert_to_one_hot(Y,class_max):
    Y = np.eye(class_max)[Y.reshape(-1)].T
    return Y