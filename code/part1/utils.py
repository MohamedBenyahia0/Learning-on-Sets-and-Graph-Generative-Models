"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    X_train=[]
    y_train=[]
    for i in range(n_train):
        num_digits=np.random.randint(1,max_train_card+1)
        digits=np.random.randint(1,max_train_card+1,size=num_digits)
        padded_digits=np.zeros(max_train_card)
        if num_digits<max_train_card:
            padded_digits[max_train_card-num_digits:]=digits
        X_train.append(padded_digits)
        y_train.append(sum(digits))
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    n_test = 200000
    max_test_card = 100
    X_test=[]
    X_test_grouped=[]
    y_test_grouped=[]
    y_test=[]
    
    for i in range(n_test):
        num_digits = 5 + (i // 10000) * 5
        num_digits = min(num_digits, max_test_card)
        
        digits=np.random.randint(1,11,size=num_digits)
        X_test.append(digits)
        y_test.append(sum(digits))
        if (i + 1) % 10000 == 0:
            
            X_group_matrix = np.array([sample[:num_digits] for sample in X_test[-10000:]])
            X_test_grouped.append(X_group_matrix)
            y_test_grouped.append(np.array([y for y in y_test[-10000:]]))
       
    
    ##################

    return X_test_grouped, y_test_grouped
