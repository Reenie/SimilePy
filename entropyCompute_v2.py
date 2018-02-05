import numpy as np

import VectorSpace_simile
import main as main
import operator
import timeit


class EntropyCompute_v2:
    def __init__(self):
        s = EntropyCompute_v2
        start = timeit.default_timer()
        s.calcAndPrintEntropyOfAllFiles_v2('self')
        s.calcAndPrintEntropyOfAllFiles('self')
        print(str(round(timeit.default_timer() - start, 3)) + " sec")

    def calcAndPrintEntropyOfAllFiles(self):
        s = EntropyCompute_v2
        for file in main.filenames:
            #print(file)
            entropy, normEntropy, KeyPercentageFreq, featureNames = s.calcEntropy(self, file)
            print("\r\n\r\n" + file)
            print('%-14s%-9s%-12s' % ("Feature", "Entropy", "Frequency of values"))
            for a, b, c, d in zip(entropy, normEntropy, KeyPercentageFreq, featureNames):
                #cc = s.sortedArrayOfHashtable(self, c)
                #cc = s.sort(key=lambda x: x[1])
                cc = sorted(c, key=lambda x:(-x[2]))
                print('\r%-14s%-9s%-12s' % (d, a, cc))
        entropy, normEntropy, KeyPercentageFreq, featureNames = s.calcEntropy_inWholeDataset(self, main.filenames)
        print("\r\n\r\nEntropy in Whole Dataset")
        print('%-14s%-9s%-12s' % ("Feature", "Entropy", "Frequency of values"))
        for a, b, c, d in zip(entropy, normEntropy, KeyPercentageFreq, featureNames):
            #cc = s.sortedArrayOfHashtable(self, c)
            cc = sorted(c, key=lambda x:(-x[2]))
            print('\r%-14s%-9s%-12s' % (d, a, cc))


    def calcAndPrintEntropyOfAllFiles_v2(self):
        s = EntropyCompute_v2
        count_file = 0
        flag_ofHeading = 0
        for file in main.filenames:
            count_file = count_file + 1
            entropy, normEntropy, KeyPercentageFreq, featureNames = s.calcEntropy(self, file)
            #normEntropyPerFile.append((count_file, normEntropy))
            if flag_ofHeading == 0:
                print("\r\n")
                featureNames_tuple = tuple(featureNames)
                #print(featureNames_tuple)
                print('%-4s%-7s%-10s%-9s%-11s%-6s%-6s%-6s%-6s%-13s%-6s%-7s%-9s%-6s%-6s%-6s%-5s%-6s%-5s' % (('Sim',) + featureNames_tuple))
                flag_ofHeading = 1
            ne_tuple = tuple(entropy)
            print('\r%-4s%-7s%-10s%-9s%-11s%-6s%-6s%-6s%-6s%-13s%-6s%-7s%-9s%-6s%-6s%-6s%-5s%-6s%-5s' % ((count_file,) + ne_tuple))
        entropy, normEntropy, KeyPercentageFreq, featureNames = s.calcEntropy_inWholeDataset(self, main.filenames)
        normEntropy_tuple = tuple(entropy)
        print('\r%-4s%-7s%-10s%-9s%-11s%-6s%-6s%-6s%-6s%-13s%-6s%-7s%-9s%-6s%-6s%-6s%-5s%-6s%-5s' % (('All',) + normEntropy_tuple))




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
        key_percentage_freq_array = []
        for vector in vectorSpaceT:
            normalized_entropy = ec.normalizedEntropy(self, vector)
            entropy, key_percentage_freq = ec.entropy(self, vector)
            entropy_array.append(entropy)
            key_percentage_freq_array.append(key_percentage_freq)
            normalized_entropy_array.append(normalized_entropy)
        return entropy_array, normalized_entropy_array, key_percentage_freq_array,  featurenames


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
        key_percentage_freq_array = []
        for vector in vectorSpaceT:
            normalized_entropy = ec.normalizedEntropy(self, vector)
            entropy, key_percentage_freq = ec.entropy(self, vector)
            entropy_array.append(entropy)
            key_percentage_freq_array.append(key_percentage_freq)
            normalized_entropy_array.append(normalized_entropy)
        return entropy_array, normalized_entropy_array, key_percentage_freq_array, featurenames


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
            normalized_entropy = data_entropy/np.math.log2(count_hashKeys)
        return round(normalized_entropy, 2)


    # entropy of the given vector
    def entropy(self, vector):
        s = EntropyCompute_v2
        key_freq = {} #hashtable
        key_percentage_frequent_array = []
        data_entropy = 0.0
        for v in vector:
            if s.hasKey(self, key_freq, v):
                key_freq[v] += 1
            else:
                key_freq[v] = 1
        sum_of_freq = s.SumOfHashtableValues(self, key_freq)
        for freq in key_freq.values():
            data_entropy += (-freq / sum_of_freq) * np.math.log2(freq / sum_of_freq)
        for k in key_freq.keys():
            key_percentage = 100*key_freq[k]/sum_of_freq
            key_percentage_frequent_array.append((k, str(round(key_percentage, 1))+'%', key_freq[k]))
        ########################
        #print(key_percentage_frequent_array)
        return round(data_entropy, 2), key_percentage_frequent_array


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






#EntropyCompute_v2()
