import numpy as np
import scipy

import main as main
import VectorSpace_simile as VSsimile
import EntropyCompute_v2 as EC
from scipy.stats.stats import pearsonr


class Variability_Measure:

    def __init__(self):
        s = Variability_Measure
        s.print_syntactic_and_semantic_entropy(self)


    def print_syntactic_and_semantic_entropy(self):
        s = Variability_Measure
        distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectorsPerSimile = \
            s.distinctNumericalVectorSpace_syntactic(self)
        print(numericalFeature_names)
        a, b, n = s.distinctNumericalVectorSpace_inWholeDataset_syntactic(self)
        print("\r\n\r\n")
        print('%-36s%-19s%-19s' % ("File Name", "Syntactic_Entropy", "Semantic_Entropy"))
        semanticEntropy_perFile = s.semanticEntropy_perSimile(self)
        #print(semanticEntropy_perFile)
        syntactic_entropy_list = []
        semantic_entropy_list = []
        for syntactic, semantic_entropy in zip(numberOfDistinctVectorsPerSimile, semanticEntropy_perFile):
            syntactic_entropy, normalized_entropy = s.entropyOfDistinctVectors(self, syntactic[4], syntactic[2])
            print('\r%-36s%-19s%-19s' % (syntactic[0], syntactic_entropy, semantic_entropy[1]))
            syntactic_entropy_list.append(syntactic_entropy)
            semantic_entropy_list.append(semantic_entropy[1])
            if syntactic[0] != semantic_entropy[0]:
                print(semantic_entropy[0] + " != " + syntactic[0])
        semantic_entropy_inWhole = s.semanticEntropy_inWholeDataset(self)
        syntactic_entropy_inwhole, normalized_entropy = s.entropyOfDistinctVectors(self, n[4], n[2])
        print('\r%-36s%-19s%-19s' % (n[0], syntactic_entropy_inwhole, semantic_entropy_inWhole[0][1]))
        #syntactic_entropy_list.append(syntactic_entropy_inwhole)
        #semantic_entropy_list.append(semantic_entropy_inWhole[0][1])
        #print("\r\n")
        #print(syntactic_entropy_list)
        #print(semantic_entropy_list)
        pearson_correlation = pearsonr(semantic_entropy_list, syntactic_entropy_list)
        #print(pearsonr([1, 2, 3, 4, 5, 6], [-6, -5, -4, -3, -2, -1]))
        print("\nPearson_correlation(syntactic_entropy, semantic_entrophy) = " + str(round(pearson_correlation[0],4)) +
              ",  p-value = " + str(round(pearson_correlation[1],4)))


    def semanticEntropy_perSimile(self):
        s = Variability_Measure
        vss = VSsimile.VectorSpace_simile
        vss.attrForVectorSpace = vss.semantic_attr
        #entropyCompute = entropyCompute_v2
        semanticEntropy_perFile = []
        for file in main.filenames:
            # print(file)
            entropy, normEntropy, KeyPercentageFreq, featureNames = EC.EntropyCompute_v2.calcEntropy(self, file)
            for a, b, c, d in zip(entropy, normEntropy, KeyPercentageFreq, featureNames):
                #print(d)
                if d == "SEMANTICS":
                    semanticEntropy_perFile.append((file, a))
        return semanticEntropy_perFile


    def semanticEntropy_inWholeDataset(self):
        s = Variability_Measure
        #vss = VSsimile.VectorSpace_simile
        #vss.attrForVectorSpace = vss.semantic_attr
        #entropyCompute = entropyCompute_v2.EntropyCompute_v2
        semanticEntropy_inWholeDataset = []
        entropy, normEntropy, KeyPercentageFreq, featureNames = EC.EntropyCompute_v2.calcEntropy_inWholeDataset(self, main.filenames)
        #print(featureNames)
        for a, b, c, d in zip(entropy, normEntropy, KeyPercentageFreq, featureNames):
            if d == "SEMANTICS":
                semanticEntropy_inWholeDataset.append(("All", a))
        return semanticEntropy_inWholeDataset




    # the entropy of distinct vectors
    def entropyOfDistinctVectors(self, vectorOfDistinctVectors, allVectors):
        entropy = 0.0
        normalized_entropy = 0.0
        for v in vectorOfDistinctVectors:
            entropy += (-len(v) / allVectors) * np.math.log2(len(v) / allVectors)
        if (len(vectorOfDistinctVectors) != 1):
            normalized_entropy = entropy / np.math.log2(len(vectorOfDistinctVectors))
        return round(entropy, 2), round(normalized_entropy, 2)

    # test
    def log(self):
        return -7 / 16 * np.math.log2(7 / 16) - 2 / 16 * np.math.log2(2 / 16) - 4 / 16 * np.math.log2(
            4 / 16) - 3 / 16 * np.math.log2(1 / 16)



    def distinctNumericalVectorSpace_syntactic(self):
        vss = VSsimile.VectorSpace_simile
        vss.attrForVectorSpace = vss.syntactic_attr  # attributes without GENDER, MWE_TYPE and SEMANTICS
        distinct_numericalVectorSpace = []
        numberOfDistinctVectorsPerSimile = []  # vectros with the excel cell number of the same vectors per simile
        numericalFeature_names = []
        for filename in main.filenames:
            numericalFeature_names, numericalVS = vss.numericalVectorSpace(self, [filename])
            # numberoOfVectors_per_simile.append((filename, len(numericalVS)))
            count = 0
            distinct_numericalVectorSpace_perSimile = []
            excelNumber_of_theSameVectors = []
            for v in numericalVS:
                if v[3:] not in distinct_numericalVectorSpace_perSimile:
                    count = count + 1
                    distinct_numericalVectorSpace_perSimile.append(v[3:])
                    distinct_numericalVectorSpace.append(v)
            for e in distinct_numericalVectorSpace_perSimile:
                theSameVectors = []
                for v in numericalVS:
                    if e == v[3:]:
                        theSameVectors.append(v[1])
                numberOfFile = numericalVS[1][0]
                excelNumber_of_theSameVectors.append(theSameVectors)
            numberOfDistinctVectorsPerSimile.append((filename, count, len(numericalVS),
                                                     str(round(count * 100 / len(numericalVS), 1)) + '%',
                                                     excelNumber_of_theSameVectors))
        # print(numericalFeature_names)
        # print(numberOfDistinctVectorsPerSimile)
        # print(distinct_numericalVectorSpace)
        # print(excelNumber_of_theSameVectors)
        return distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectorsPerSimile

    def distinctNumericalVectorSpace_inWholeDataset_syntactic(self):
        vss = VSsimile.VectorSpace_simile
        vss.attrForVectorSpace = vss.syntactic_attr  # attributes without GENDER, MWE_TYPE and SEMANTICS
        distinct_numericalVectorSpace = []
        distinct_vectors = []
        # numberOfDistinctVectors = []
        numericalFeature_names, numericalVS = vss.numericalVectorSpace(self, main.filenames, gender=vss.numOfGenders)
        count = 0
        for v in numericalVS:
            if v[3:] not in distinct_vectors:
                count = count + 1
                distinct_vectors.append(v[3:])
                distinct_numericalVectorSpace.append(v)
        simile_and_excelNumber_of_theSameVectors = []  # vector of (number_of_simile, excel_number)
        # simile and excel Numbers of the Same Vectors
        for d in distinct_numericalVectorSpace:
            temp = []
            for v in numericalVS:
                if v[3:] == d[3:]:
                    temp.append((v[0], v[1]))
            simile_and_excelNumber_of_theSameVectors.append(temp)
        numberOfDistinctVectors = ('In Whole Dataset', count, len(numericalVS),
                                   str(round(count * 100 / len(numericalVS), 1)) + '%',
                                   simile_and_excelNumber_of_theSameVectors)
        ###################
        # print(numberOfDistinctVectors[2])
        # print(numberOfDistinctVectors[4])
        return distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectors





        # per distinct vector --> simile and number of vectors

    def groups_of_DistinctVector(self, simile_and_excelNumber_of_theSameVectors):
        s = Variability_Measure
        distinctVectors = []
        for vector in simile_and_excelNumber_of_theSameVectors:
            hash = {}
            for tuple in vector:
                if s.hasKey(self, hash, tuple[0]):
                    hash[tuple[0]] += 1
                else:
                    hash[tuple[0]] = 1
            numOfHashValues = s.SumOfHashtableValues(self, hash)
            temp_list = []
            for key in hash.keys():
                temp_list.append((key, hash[key]))
            temp_list.sort(key=lambda x: -x[1])
            temp_tuple = (numOfHashValues, temp_list)
            distinctVectors.append(temp_tuple)
        distinctVectors.sort(key=lambda x: -x[0])
        ############
        # print(distinctVectors)
        return distinctVectors

    def hasKey(self, hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0

    def SumOfHashtableValues(self, hashTable):
        sum = 0
        for k in hashTable.keys():
            sum += hashTable[k]
        return sum


Variability_Measure()
