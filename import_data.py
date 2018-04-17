
import numpy as np
import random
import re
import copy




def get_data(which):
    with open("data/"+ str(which) +"_data.csv","r", encoding="utf-8") as f_data_in:
            lines = f_data_in.readlines()
            # print(type(lines))
            dataset = list()
            for line in lines:
                line = re.sub("\s+", ",", line.strip())
                parts = line.split(",")
                dataset.append(parts)
            dataset = np.array(dataset, dtype=np.float64)

    with open("data/"+ str(which) +"_targets.csv","r", encoding="utf-8") as f_target_in:
            lines = f_target_in.readlines()
            targets = list()
            for line in lines:
                targets.append(line)
            targets = np.array(targets, dtype=np.int64)


    x_data = dataset
    y_data = np.atleast_2d(targets).T


    print("dataset :")
    print(x_data.shape)
    print("targets :")
    print(y_data.shape)



    """ max-min normalizing each feature (not rows!) """
    for col in range(x_data.shape[1]):
        maxx = max(x_data[:, col])
        minn = min(x_data[:, col])
        # if((maxx - minn) == 0):
        #     x_data[:, col] = 0
        #     x_data[0, col] = 0.5
        #     maxx = max(x_data[:, col])
        #     minn = min(x_data[:, col])

        # if((maxx - minn) == 0):
        #     x_data = np.delete(x_data, np.array([col]), 1)
        # if((maxx - minn) != 0):
        x_data[:, col] = 2*(x_data[:, col] - minn)/(maxx - minn)-1
        # if(np.any(ii is None for ii in x_data[:, col])):
        #     print(ii)
        #     print(col)



    # shuffling data for fun
    idx = np.random.permutation(x_data.shape[0])
    x_data, y_data = x_data[idx], y_data[idx]

    # print("x_data : ")
    # print(x_data)
    # print("y_data : ")
    # print(y_data)

    return separate_datasets(x_data, y_data)





# if(((maxx - minn)-1.1) == 0):
#     print("hellloooo")
    # x_data[:, col] = 3*(x_data[:, col] - minn)/0.01
# else:








def separate_datasets(x, y):

    leng = len(x[:, 0])
    s = int(leng*5/10)

    samples_list = random.sample(range(0, leng), s)

    mask = np.zeros((leng), dtype=bool)
    mask[samples_list] = True

    x_data = x[mask, :]
    y_data = y[mask]

    testing_dataset = x[~mask, :]
    testing_targets = y[~mask]


    # separating val set and test :
    test_leng = len(testing_targets)
    ss = int(test_leng/2)

    samples_list = random.sample(range(0, test_leng), ss)

    mask = np.zeros((test_leng), dtype=bool)
    mask[samples_list] = True


    x_test_set = testing_dataset[mask, :]
    y_test_set = testing_targets[mask]

    x_val_set = testing_dataset[~mask, :]
    y_val_set = testing_targets[~mask]


    del testing_dataset
    del testing_targets


    print("x_data : " + str(x_data.shape))
    print("y_data : " + str(y_data.shape))

    print("x_test_set : " + str(x_test_set.shape))
    print("y_test_set : " + str(y_test_set.shape))

    print("x_val_set : " + str(x_val_set.shape))
    print("y_val_set : " + str(y_val_set.shape))


    classes = np.unique(y)
    for i in range(0, len(classes)):
        tags = np.zeros((len(y_data)), dtype="int64")
        bool_arr = np.equal(y_data, classes[i])
        tags[bool_arr[:,0]] = 1
        print("num (y_data) for class " + str(i) + " : " + str(np.sum(tags)))


    for i in range(0, len(classes)):
        tags = np.zeros((len(y_test_set)), dtype="int64")
        bool_arr = np.equal(y_test_set, classes[i])
        tags[bool_arr[:,0]] = 1
        print("num (y_test_set) for class " + str(i) + " : " + str(np.sum(tags)))


    for i in range(0, len(classes)):
        tags = np.zeros((len(y_val_set)), dtype="int64")
        bool_arr = np.equal(y_val_set, classes[i])
        tags[bool_arr[:,0]] = 1
        print("num (y_val_set) for class " + str(i) + " : " + str(np.sum(tags)))


    return (x_data, y_data, x_val_set, y_val_set, x_test_set, y_test_set)







