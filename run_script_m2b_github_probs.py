
'''
####################### Running Instructions ##############################################
python3 run_script_m2b_github_probs.py <train filename> <test filename> <example_samples> <class_sampling> <Candidates>

example_samples (mu): Number of examples to be taken per class ( e.g. values 1, 2, 5)

class_sampling: Sampling rate for choosing classes to sample ( e.g. 0.1, 0.01, 0.001) (Note: The minimum value for class_
sampling is set as 1 / Size of class, if user enters less than this value by default 1 class will be chosen.

Candidates (sigma): Number of candidate classes for prediction (e.g. 10, 20, 50)

UPDATED THE PREPROCESS CODE IN THIS VERSION. THIS UPDATES MAKES IT BIT FASTER AND MORE MEMROY EFFICIENT
'''
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import *
import itertools, math, codecs, sys, random, json, numpy as np, scipy.sparse
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import time
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import resource
import warnings
import os
from os.path import expanduser
from collections import Counter

def score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    maf = f1_score(y_true, y_pred, average='macro')
    print(acc)
    print(maf)
    return acc, maf


def CreateFolder(Dataset, foldername, Num_of_Samples, Sampling, num_candidates):
    timestamp = str(int(time.time()))[-4:]
    home = expanduser("~")
    folderpath = home + "/Code/multi2binary_local/Results/" + Dataset + "/Modified/" + foldername + "/" + str(
        Num_of_Samples) + "_" + str(Sampling) + "_" + str(num_candidates) + "_" + timestamp
    # folderpath = home + "/Code/multi2binary_local/Results/DMOZ/" + "Test"
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    return folderpath


def Preprocess_m2b(X, y, X_test, y_test):
    tfidf = TfidfTransformer()
    tfidf.fit(X)
    X_tfidf = tfidf.transform(X, copy=False)
    X_test_tfidf = tfidf.transform(X_test, copy=False)
    class_list = np.array(list(set(y)))
    classes_length = {}
    class_map = defaultdict(csr_matrix)
    # calculate mean vector for each class and global one for whole collection
    idfs = tfidf.idf_
    collection_vec = X.sum(axis=0)
    class_centroids = defaultdict(csr_matrix)
    #print("Total memory usage before loop : %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    ind = 0
    total_nz = 0
    indptr = 0
    for i in class_list:
        X_tfidf_class = X_tfidf[y == i]
        class_map[i] = csr_matrix(X[y == i].sum(axis=0))
        classes_length[i] = class_map[i].sum()  ### k= 2000, t = 6.71s, mem= 144MB
        temp = csr_matrix((X_tfidf_class.sum(axis=0)) / (X_tfidf_class.shape[0]))  ### k= 2000, t = 11.45s, mem = 147MB
        total_nz += temp.nnz
        class_centroids[i] = temp  ### k= 2000, t = 11.45s, mem = 173MB

    rows = np.empty(total_nz)
    cols = np.empty(total_nz)
    dats = np.empty(total_nz)

    for i in class_list:
        temp = class_centroids[i]
        c = temp.indices
        d = temp.data
        rows[indptr: indptr + len(c)] = ind
        cols[indptr: indptr + len(c)] = c
        dats[indptr: indptr + len(c)] = d
        indptr = indptr + len(c)
        ind += 1

    temp_coo = coo_matrix((dats, (rows, cols)), shape=(len(class_list), X.shape[1]), dtype="float16")
    class_centroids_arr = csr_matrix(temp_coo)
    return X_tfidf, X_test_tfidf, tfidf, idfs, collection_vec, class_list, class_map, classes_length, class_centroids, class_centroids_arr


def extractFeatures(vecy_i, vec_y, centroid_distance, S_vec, len_collection, idfs, length_y, D, avg_length):
    # vecy_i : current vector i -> feature vector
    # vec_y : vector of class y -> mean vector of the respective class
    # S_vec : vector of collection -> scalar sum of everything in the train set
    # idfs : inverse document frequencies
    # avg_len : average length of classes
    # X_tfidf_class: tfidf values of all vectors in the class in training data
    # X_tfidf_vec: tfidf vector of current example
    # start = time.time()
    x = [0] * D
    inter = vecy_i.multiply(vec_y).nonzero()
    vec_y_part = vec_y.toarray()[0, inter[1]]

    x[0] = np.log(1.0 + vec_y_part).sum()  # f0
    x[1] = np.log(1.0 + len_collection / S_vec[0, inter[1]]).sum()  # f1
    x[2] = idfs[inter[1]].sum()  # f2
    x[3] = np.log(1.0 + vec_y_part / length_y).sum()  # f4
    x[4] = np.log(1.0 + (np.multiply(idfs[inter[1]], vec_y_part)) / length_y).sum()  # f5
    x[5] = np.log(1.0 + (vec_y_part / length_y) * len_collection / S_vec[0, inter[1]]).sum()  # f6
    x[6] = len(inter[1])  # f7
    x[7] = centroid_distance  # f8
    x[8] = len(inter[1]) / vec_y.getnnz()  # f13 improves a bit
    ## BM 25
    x[9] = np.log(1.0 + idfs[inter[1]] * (2.0 * vec_y_part / (vec_y_part + \
                                                              (0.25 + 0.75 * length_y / avg_length)))).sum()  # bm25

    #print(x)
    return x


'''
This method runs on a single core during the reduction phase.
'''
def Reduction(X, y, tfidf, class_map, sampling, num_sample, classes_length,
              collection_vec, D, class_centroids):
    len_collection = collection_vec.sum()
    X_trans = []
    Y_tr = []
    idfs = tfidf.idf_
    # centroid_distances = np.load(home + "/Code/multi2binary_local/Centroids/" + Dataset + "/" + foldername + "/centroid_distances.npy", mmap_mode='r')

    counter = Counter(y)
    class_prob = [counter[i] / len(y) for i in class_list]
    selection = np.random.choice(class_list, len(class_list) * num_sample, p=class_prob)
    new_dist = Counter(selection)
    for x in (counter.keys() - new_dist.keys()):
        new_dist[x] = 1
    #print(len(new_dist.keys()))
    #print(new_dist)
    count_red = 0
    avg_length = len_collection / float(len(classes_length))
    for i in new_dist.keys():
        X_class = X[y == i]
        len_class = X_class.shape[0]
        # centroid_distances_class = centroid_distances[:, y == i]
        new_num_sample = new_dist[i]
        if len_class > num_sample:
            sample = np.random.choice(np.array(range(len_class)), new_num_sample)
            X_class_sample = lil_matrix(X_class[sample])
            # centroid_distances_sample = centroid_distances_class[:, sample]
        else:
            X_class_sample = lil_matrix(X_class[:])
            # centroid_distances_sample = centroid_distances_class[:, :]
        '''
        sample = [0]
        X_class_sample = lil_matrix(X_class[sample])
        centroid_distances_sample = centroid_distances_class[:, sample]
        '''
        centroid_distances_outer = cosine_distances(class_centroids[i], X_class_sample)[0]
        for j in range(0, X_class_sample.shape[0]):
            x_yi = extractFeatures(X_class_sample.getrowview(j), class_map[i], centroid_distances_outer[j],
                                   collection_vec, len_collection, idfs, classes_length[i], D, avg_length)
            count_red += 1
            choice = int(np.ceil(len(class_list) * sampling))
            # print("Sampled choice is %d"% choice)
            class_list_sampled = np.random.choice(class_list, choice)
            for klasse in class_list_sampled:  # can be replaced by iteritems()
                # klasse = class_list[ind]
                '''
                if random.random() > sampling:
                    continue
                '''
                vec_y = class_map[klasse]
                x_k = extractFeatures(X_class_sample.getrowview(j), vec_y,
                                      cosine_distances(class_centroids[klasse], X_class_sample.getrowview(j))[0][0],
                                      collection_vec, len_collection, idfs, classes_length[klasse], D, avg_length)
                count_red += 1
                if i > klasse:
                    x_trans = np.subtract(x_yi, x_k)
                    y_tr = 1
                elif i < klasse:
                    x_trans = np.subtract(x_k, x_yi)
                    y_tr = -1
                else:
                    continue
                X_trans.append(x_trans.tolist())
                Y_tr.append(y_tr)
                # ind += 1

    #np.save(folderpath + "/train", np.array(X_trans))
    #np.save(folderpath + "/train_label", np.array(Y_tr))
    scaler = preprocessing.MinMaxScaler().fit(X_trans)
    X_trans_norm = scaler.transform(X_trans)
    '''
    with open("./train_binary", "w") as f:
        for x_i, y_i in zip(X_trans_norm, Y_tr):
            count = 1
            f.write(str(y_i))
            for i in x_i:
                f.write(" ".join([" " + "{}:{}".format(count, i)]))
                count += 1
            f.write("\n")
    '''
    return scaler, X_trans_norm, Y_tr


'''
This method runs on a single core during the reduction phase.
'''

def Learn(binary_features, binary_labels):
    #binary_features, binary_labels = load_svmlight_file("./train_binary")
    clf = LinearSVC(C=0.01)
    clf.fit(binary_features, binary_labels)
    weights = clf.coef_[0]
    #np.save("./weights", weights)
    return weights


def Prediction(X_test_tfidf, class_list, X_test, class_map, collection_vec, idfs, classes_length, D, scaler):
    nbrs = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=num_candidates).fit(class_centroids_arr)
    # print("before")
    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # NN = nbrs.kneighbors(X_test_tfidf, return_distance=True)
    # print("after")
    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    len_collection = collection_vec.sum()
    avg_length = len_collection / float(len(classes_length))
    y_pred = []
    X_test_lil = X_test.tolil()
    test = []
    rowList = np.arange(X_test_tfidf.shape[0])
    rowRangeList = []
    block_size = 1000
    partition = math.ceil(len(rowList) / block_size)
    #print(partition)
    for i in range(partition):
        if (i < partition - 1):
            rowRangeList.append(rowList[i * block_size:(i + 1) * block_size])
        else:
            rowRangeList.append(rowList[i * block_size:])

    y_pred = []
    test = []
    for block in rowRangeList:
        # candidate_set = [class_list[x] for x in NN[1][i]]
        # print(candidate_set)
        NN_block = nbrs.kneighbors(X_test_tfidf[block], return_distance=True)

        for indx in range(len(block)):
            i = block[indx]
            candidate_set = [class_list[x] for x in NN_block[1][indx]]
            # print(candidate_set)
            ind = 0
            X_test_candidate = np.zeros((len(candidate_set), D))

            for cl in range(len(candidate_set)):
                klasse = candidate_set[cl]
                vec_y = class_map[klasse]
                # print(NN[0][i][cl])
                # print(NN_current[0][0][cl])
                X_test_candidate[ind] = extractFeatures(X_test_lil.getrow(i), vec_y, NN_block[0][indx][cl],
                                                        collection_vec, len_collection, idfs, classes_length[klasse], D,
                                                        avg_length)
                # X_test_candidate[ind] = extractFeatures(X_test_lil.getrow(i), vec_y, NN_current[0][0][cl], collection_vec,
                #                                        len_collection, idfs, classes_length[klasse], D, avg_length)
                ind += 1
            X_test_k = scaler.transform(X_test_candidate)
            y_pred.append(
                candidate_set[
                    np.argmax(np.array([np.dot(weights, X_test_k[p]) for p in range(len(X_test_k))]))].tolist())
            test.append(np.argmax(np.array([np.dot(weights, X_test_k[p]) for p in range(len(X_test_k))])))

    # print(y_pred)
    #print(np.count_nonzero(test), X_test_tfidf.shape[0])

    return y_pred


########################################################################################################################
################################################# RUN CODE #############################################################
########################################################################################################################

if __name__ == "__main__":
    ##
    start_code = time.time()
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    num_samples = int(sys.argv[3])
    sampling_rate = float(sys.argv[4])
    num_candidates = int(sys.argv[5])
    feature_size = 10

    print(" <<<<<<<<<<<<<<<<  START  Algorithm  >>>>>>>>>>>>>>>")
    print("Num_sample = %f, Sampling = %f, Num_candidates = %d" % (num_samples, sampling_rate, num_candidates))
    warnings.filterwarnings("ignore")

    ############################################## Load Data ###########################################################
    start_load = time.time()
    X, y, X_test, y_test = load_svmlight_files((train_file, test_file),dtype='float32')  # Load tf vectors
    stop_load = time.time()
    print("Time for loading dataset: %d seconds" % (stop_load - start_load))

    ############################################## Preprocess Data #####################################################
    start_preprocess = time.time()
    X_tfidf, X_test_tfidf, tfidf, idfs, collection_vec, class_list, class_map, classes_length, class_centroids, \
    class_centroids_arr = Preprocess_m2b( X, y, X_test, y_test)
    stop_preprocess = time.time()
    print("Time for preprocessing: %d seconds" % (stop_preprocess - start_preprocess))

    ##############################################  Reduction Step #####################################################

    start_reduction = time.time()
    scaler, X_binary, y_binary = Reduction(X, y, tfidf, class_map, sampling_rate, num_samples, classes_length,collection_vec, \
                       feature_size, class_centroids)
    stop_reduction = time.time()
    print("Time for binary reduction: %d seconds:" % (stop_reduction - start_reduction))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    ############################################### Learning Algorithm #################################################
    start_learn = time.time()
    weights = Learn(X_binary, y_binary)
    #print(weights)
    stop_learn = time.time()
    print("Time for learning: %d seconds:" % (stop_learn - start_learn))
    ############################################### Prediction  Step ###################################################

    start_pred = time.time()
    y_pred = Prediction(X_test_tfidf, class_list, X_test, class_map, collection_vec, idfs,  classes_length, \
                        feature_size, scaler)
    acc, maf = score(y_test, y_pred)
    stop_pred = time.time()
    print("Time for prediction: %d seconds: \n" % (stop_pred - start_pred))
    print("Total Runtime: %d seconds"%(stop_pred - start_code))
    print("Total Memory Usage: %d KB"%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
