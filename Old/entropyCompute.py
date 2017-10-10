import numpy as np

import main as main
from Old import fileToMatrix as fileToMatrix


class EntropyCompute:
    def __init__(self):
        EntropyCompute.calcAndPrintEntropyOfAllFiles('self')

    def calcAndPrintEntropyOfAllFiles(self):
        s = EntropyCompute
        for file in main.filenames:
            entropy, normEntropy, freqKey, featureNames = s.calcEntropy(file)
            print("\r\n\r\n" + file)
            print('%-28s%-12s%-20s%-12s' % ("Feature", "Entropy", "Normalized Entropy", "Frequency of values"))
            for a, b, c, d in zip(entropy, normEntropy, freqKey, featureNames):
                #print('{}\t\t{}\t{}\t{}'.format(d, a, b, c))
                print('\r%-28s%-12s%-20s%-12s' % (d, a, b, c))
        entropy, normEntropy, freqKey, featureNames = s.calcEntropy_inWholeDataset(main.filenames)
        print("\r\n\r\nEntropy in Whole Dataset")
        print('%-28s%-12s%-20s%-12s' % ("Feature", "Entropy", "Normalized Entropy", "Frequency of values"))
        for a, b, c, d in zip(entropy, normEntropy, freqKey, featureNames):
            # print('{}\t\t{}\t{}\t{}'.format(d, a, b, c))
            print('\r%-28s%-12s%-20s%-12s' % (d, a, b, c))



    def calcEntropy(filename):
        ftm = fileToMatrix.FileToMatrix
        ec = EntropyCompute
        vectorSpace, featurenames = ftm.fileToVectorSpace_from2ndColumn_usefulAttr(filename)
        vectorSpaceT = ftm.rowToColTransposition(vectorSpace)
        entropy_array = []
        normalized_entropy_array = []
        freq_key_array = []
        for vector in vectorSpaceT:
            normalized_entropy = ec.normalizedEntropy(vector)
            entropy, freq_key= ec.entropy(vector)
            entropy_array.append(entropy)
            freq_key_array.append(freq_key)
            normalized_entropy_array.append(normalized_entropy)
        return entropy_array, normalized_entropy_array, freq_key_array, featurenames


    def calcEntropy_inWholeDataset(filenames):
        ftm = fileToMatrix.FileToMatrix
        ec = EntropyCompute
        vectorSpace, featurenames = ftm.VectorSpace_InWoleDataset_fromSecondColumn_usefulAttr(filenames)
        vectorSpaceT = ftm.rowToColTransposition(vectorSpace)
        entropy_array = []
        normalized_entropy_array = []
        freq_key_array = []
        for vector in vectorSpaceT:
            normalized_entropy = ec.normalizedEntropy(vector)
            entropy, freq_key = ec.entropy(vector)
            entropy_array.append(entropy)
            freq_key_array.append(freq_key)
            normalized_entropy_array.append(normalized_entropy)
        return entropy_array, normalized_entropy_array, freq_key_array, featurenames


    # normalized entropy in [0,1]
    def normalizedEntropy(vector):
        key_freq = {}
        data_entropy = 0.0
        normalized_entropy = 0.0
        for v in vector:
            if EntropyCompute.hasKey(key_freq, v):
                key_freq[v] += 1
            else:
                key_freq[v] = 1
        sum_of_freq = EntropyCompute.SumOfHashtableValues(key_freq)
        count_hashKeys = EntropyCompute.countOfHashtableKeys(key_freq)
        for freq in key_freq.values():
            data_entropy += (-freq / sum_of_freq) * np.math.log2(freq / sum_of_freq)
        if count_hashKeys != 1:
            normalized_entropy = data_entropy/ np.math.log2(count_hashKeys)
        return round(normalized_entropy, 3)


    # entropy of the given vector
    def entropy(vector):
        key_freq = {}
        data_entropy = 0.0
        for v in vector:
            if EntropyCompute.hasKey(key_freq, v):
                key_freq[v] += 1
            else:
                key_freq[v] = 1
        sum_of_freq = EntropyCompute.SumOfHashtableValues(key_freq)
        for freq in key_freq.values():
            data_entropy += (-freq / sum_of_freq) * np.math.log2(freq / sum_of_freq)
        #print(key_freq)
        return round(data_entropy, 3), key_freq





    def countOfHashtableKeys(hashTable):
        count = 0
        for k in hashTable.keys():
            count += 1
        return count

    def SumOfHashtableValues(hashTable):
        sum = 0
        for k in hashTable.keys():
            sum += hashTable[k]
        return sum


    def hasKey(hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0

    # Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
    def gain(data, attr, target_attr):
        val_freq = {}
        subset_entropy = 0.0
        # Calculate the frequency of each of the values in the target attribute
        for record in data:
            if (val_freq.has_key(record[attr])):
                val_freq[record[attr]] += 1.0
            else:
                val_freq[record[attr]] = 1.0
        # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
        for val in val_freq.keys():
            val_prob = val_freq[val] / sum(val_freq.values())
            data_subset = [record for record in data if record[attr] == val]
            subset_entropy += val_prob * EntropyCompute.entropy(data_subset, target_attr)
        # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
        return (EntropyCompute.entropy(data, target_attr) - subset_entropy)

    def rowToColTransposition(m):
        rows = len(m)  # rows
        cols = len(m[0])  # columns
        mT = []
        for c in range(0, cols):
            mT.append([])
            for r in range(0, rows):
                mT[c].append(m[r][c])
        return mT

    def countUniqueElements(matrix):
        for i in range(0, 6):
            i = i + 1


            # it returns the max number between a and b

    def maxNumber(a, b):
        if a > b:
            return a
        return b



EntropyCompute()
#ftm = fileToMatrix.FileToMatrix
#ec.calcEntropy_ofAllFiles(main.filenames)
# EntropyCompute()


'''
    def rowToColTransposition(m):
        a = np.array(m)
        aT = np.transpose(a)
        return a

    def column(matrix, i):
        return [row[i] for row in matrix]


'''
