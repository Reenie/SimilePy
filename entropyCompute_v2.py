import numpy as np

import VectorSpace_simile
import main as main


class EntropyCompute_v2:
    def __init__(self):
        s = EntropyCompute_v2
        s.calcAndPrintEntropyOfAllFiles('self')

    def calcAndPrintEntropyOfAllFiles(self):
        s = EntropyCompute_v2
        for file in main.filenames:
            print(file)
            entropy, normEntropy, freqKey, featureNames = s.calcEntropy(self, file)
            print("\r\n\r\n" + file)
            print('%-28s%-12s%-20s%-12s' % ("Feature", "Entropy", "Normalized Entropy", "Frequency of values"))
            for a, b, c, d in zip(entropy, normEntropy, freqKey, featureNames):
                cc = s.sortedArrayOfHashtable(self, c)
                print('\r%-28s%-12s%-20s%-12s' % (d, a, b, cc))
        entropy, normEntropy, freqKey, featureNames = s.calcEntropy_inWholeDataset(self, main.filenames)
        print("\r\n\r\nEntropy in Whole Dataset")
        print('%-28s%-12s%-20s%-12s' % ("Feature", "Entropy", "Normalized Entropy", "Frequency of values"))
        for a, b, c, d in zip(entropy, normEntropy, freqKey, featureNames):
            cc = s.sortedArrayOfHashtable(self, c)
            print('\r%-28s%-12s%-20s%-12s' % (d, a, b, cc))






    def calcEntropy(self, filename):
        vs3 = VectorSpace_simile.VectorSpace_simile
        ec = EntropyCompute_v2
        vectorSpace, featurenames = vs3.featureMatrix("self", [filename], attrForVectorSpace=vs3.attrForVectorSpace)
        featurenames = np.array(featurenames)
        featurenames = featurenames[3:]
        vectorSpace = np.array(vectorSpace)
        ################3
        #print(vectorSpace)
        vectorSpaceT = vs3.rowToColTransposition("self", vectorSpace[0:, 3:])
        entropy_array = []
        normalized_entropy_array = []
        freq_key_array = []
        for vector in vectorSpaceT:
            normalized_entropy = ec.normalizedEntropy(self, vector)
            entropy, freq_key= ec.entropy(self, vector)
            entropy_array.append(entropy)
            freq_key_array.append(freq_key)
            normalized_entropy_array.append(normalized_entropy)
        return entropy_array, normalized_entropy_array, freq_key_array, featurenames


    def calcEntropy_inWholeDataset(self, filenames):
        vs3 = VectorSpace_simile.VectorSpace_simile
        ec = EntropyCompute_v2
        vectorSpace, featurenames = vs3.featureMatrix("self", filenames, attrForVectorSpace=vs3.attrForVectorSpace)
        featurenames = np.array(featurenames)
        featurenames = featurenames[3:]
        vectorSpace = np.array(vectorSpace)
        vectorSpaceT = vs3.rowToColTransposition("self", vectorSpace[0:, 3:])
        entropy_array = []
        normalized_entropy_array = []
        freq_key_array = []
        for vector in vectorSpaceT:
            normalized_entropy = ec.normalizedEntropy(self, vector)
            entropy, freq_key = ec.entropy(self, vector)
            entropy_array.append(entropy)
            freq_key_array.append(freq_key)
            normalized_entropy_array.append(normalized_entropy)
        return entropy_array, normalized_entropy_array, freq_key_array, featurenames


    # normalized entropy in [0,1]
    def normalizedEntropy(self, vector):
        s = EntropyCompute_v2
        key_freq = {}
        data_entropy = 0.0
        normalized_entropy = 0.0
        for v in vector:
            if s.hasKey(self, key_freq, v):
                key_freq[v] += 1
            else:
                key_freq[v] = 1
        sum_of_freq = s.SumOfHashtableValues(self, key_freq)
        count_hashKeys = s.countOfHashtableKeys(self, key_freq)
        for freq in key_freq.values():
            data_entropy += (-freq / sum_of_freq) * np.math.log2(freq / sum_of_freq)
        if count_hashKeys != 1:
            normalized_entropy = data_entropy/ np.math.log2(count_hashKeys)
        return round(normalized_entropy, 3)


    # entropy of the given vector
    def entropy(self, vector):
        s = EntropyCompute_v2
        key_freq = {}
        data_entropy = 0.0
        for v in vector:
            if s.hasKey(self, key_freq, v):
                key_freq[v] += 1
            else:
                key_freq[v] = 1
        sum_of_freq = s.SumOfHashtableValues(self, key_freq)
        for freq in key_freq.values():
            data_entropy += (-freq / sum_of_freq) * np.math.log2(freq / sum_of_freq)
        #print(key_freq)
        return round(data_entropy, 3), key_freq


    def sortedArrayOfHashtable(self, hashtable={}):
        sorted_array = []
        for k in hashtable.keys():
            sorted_array.append((k, hashtable[k]))
        sorted_array.sort(key=lambda tup: tup[1], reverse=True)
        return sorted_array



    def countOfHashtableKeys(self, hashTable):
        count = 0
        for k in hashTable.keys():
            count += 1
        return count

    def SumOfHashtableValues(self, hashTable):
        sum = 0
        for k in hashTable.keys():
            sum += hashTable[k]
        return sum


    def hasKey(self, hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0

    # it returns the max number between a and b
    def maxNumber(self, a, b):
        if a > b:
            return a
        return b


'''
    # Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
    def gain(self, data, attr, target_attr):
        s = EntropyCompute_v2
        val_freq = {}
        subset_entropy = 0.0
        # Calculate the frequency of each of the values in the target attribute
        for record in data:
            if (val_freq.has_key(self, record[attr])):
                val_freq[record[attr]] += 1.0
            else:
                val_freq[record[attr]] = 1.0
        # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
        for val in val_freq.keys():
            val_prob = val_freq[val] / sum(val_freq.values())
            data_subset = [record for record in data if record[attr] == val]
            subset_entropy += val_prob * s.entropy(self, data_subset)
        # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
        return (s.entropy(self, data) - subset_entropy)

    def rowToColTransposition(self, m):
        rows = len(m)  # rows
        cols = len(m[0])  # columns
        mT = []
        for c in range(0, cols):
            mT.append([])
            for r in range(0, rows):
                mT[c].append(m[r][c])
        return mT

    def countUniqueElements(self, matrix):
        for i in range(0, 6):
            i = i + 1
'''





EntropyCompute_v2()
