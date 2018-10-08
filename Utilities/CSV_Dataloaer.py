from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split


def csv_dataloader(input_file):

    X = []
    y = []
    with open(input_file,'r') as f_in:
        line = f_in.readline()
        while line:
            if line.startswith('Flow'):
                line =f_in.readline()
            line_arr= line.split(',')
            X.append(line_arr[7:12])
            if line_arr[-1] =='2\n':
                y.append('1')
            else:
                y.append('0')

            line = f_in.readline()

    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = int)
    print(Counter(y))

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    return (X_train,y_train),(X_val,y_val),(X_test,y_test)

