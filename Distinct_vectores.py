import numpy as np
import main as main
import VectorSpace_simile as VSsimile

class Distinct_vectors:

    #txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = Distinct_vectors
        #vectorOfDistinctVectors = [['4', '7', '10', '19', '22', '25', '46'],['13', '16'], ['28', '31', '34', '37'],['40'], ['43'], ['49']   ]
        #a = s.entropyOfDistinctVectors(self, vectorOfDistinctVectors, 16)
        #print(a)
        #print(s.log(self))
        #s.distinctNumericalVectorSpace(self)
        s.printStatistics_of_DistinctVectors(self)


    def printStatistics_of_DistinctVectors(self):
        s = Distinct_vectors
        distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectorsPerSimile = s.distinctNumericalVectorSpace(self)
        print(numericalFeature_names)
        a, b, n = s.distinctNumericalVectorSpace_inWholeDataset(self)
        print("\r\n\r\n")
        #print('%-35s%-19s%-14s%-13s%-10s%-10s' % ("File Name", "Distinct Vectors", "All Vectors", "Percentage", "Entropy", "Normalized entropy"))
        print('%-35s%-19s%-14s%-13s%-10s' % ("File Name", "Distinct Vectors", "All Vectors", "Percentage", "Entropy"))
        for e in numberOfDistinctVectorsPerSimile:
            entropy, normalized_entropy = s.entropyOfDistinctVectors(self, e[4], e[2])
            #print('\r%-35s%-19s%-14s%-13s%-10s%-10s' % (e[0], e[1], e[2], e[3], entropy, normalized_entropy))
            print('\r%-35s%-19s%-14s%-13s%-10s' % (e[0], e[1], e[2], e[3], entropy))
        entropy, normalized_entropy = s.entropyOfDistinctVectors(self, n[4], n[2])
        #print('\r%-35s%-19s%-14s%-13s%-10s%-10s' % (n[0], n[1], n[2], n[3], entropy, normalized_entropy))
        print('\r%-35s%-19s%-14s%-13s%-10s' % (n[0], n[1], n[2], n[3], entropy))
        print("\r\n\r\n")

        print("Groups Of Distinct Vectors:\r\n")
        groups_Of_DistinctsVectors = s.groups_of_DistinctVector(self, n[4])
        for v in groups_Of_DistinctsVectors:
            print(v)
            print('\n')
        print("\r\n\r\n")
        print("The excel cell numbers of the same feature vectors per simile:\r\n")
        for e in numberOfDistinctVectorsPerSimile:
            print(e[0])
            for ee in e[4]:
                print(ee)
            print("\r\n\r\n")
        print("In Whole Dataset: The simile and excel cell numbers of the same feature vectors:\r\n")
        for ee in n[4]:
            print(ee)
            print("\r\n")



    #the entropy of distinct vectors
    def entropyOfDistinctVectors(self, vectorOfDistinctVectors, allVectors):
        entropy = 0.0
        normalized_entropy = 0.0
        for v in vectorOfDistinctVectors:
            entropy += (-len(v) / allVectors) * np.math.log2(len(v) / allVectors)
        if (len(vectorOfDistinctVectors) != 1):
            normalized_entropy = entropy / np.math.log2(len(vectorOfDistinctVectors))
        return round(entropy,2) , round(normalized_entropy,2)


    #test
    def log(self):
        return  -7/16*np.math.log2(7/16) - 2/16*np.math.log2(2/16) - 4/16*np.math.log2(4/16) - 3/16*np.math.log2(1/16)



    def distinctNumericalVectorSpace(self):
        vss = VSsimile.VectorSpace_simile
        distinct_numericalVectorSpace = []
        numberOfDistinctVectorsPerSimile = [] # vectros with the excel cell number of the same vectors per simile
        numericalFeature_names = []
        for filename in main.filenames:
            numericalFeature_names, numericalVS = vss.numericalVectorSpace(self, [filename])
            #numberoOfVectors_per_simile.append((filename, len(numericalVS)))
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
            numberOfDistinctVectorsPerSimile.append((filename, count, len(numericalVS), str(round(count*100/len(numericalVS),1))+'%',  excelNumber_of_theSameVectors))
        #print(numericalFeature_names)
        #print(numberOfDistinctVectorsPerSimile)
        #print(distinct_numericalVectorSpace)
        #print(excelNumber_of_theSameVectors)
        return distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectorsPerSimile


    def distinctNumericalVectorSpace_inWholeDataset(self):
        vss = VSsimile.VectorSpace_simile
        distinct_numericalVectorSpace = []
        distinct_vectors = []
        #numberOfDistinctVectors = []
        numericalFeature_names, numericalVS = vss.numericalVectorSpace(self, main.filenames, gender=vss.numOfGenders)
        count = 0
        for v in numericalVS:
            if v[3:] not in distinct_vectors:
                count = count + 1
                distinct_vectors.append(v[3:])
                distinct_numericalVectorSpace.append(v)
        simile_and_excelNumber_of_theSameVectors = [] #vector of (number_of_simile, excel_number)
        #simile and excel Numbers of the Same Vectors
        for d in distinct_numericalVectorSpace:
            temp = []
            for v in numericalVS:
                if v[3:] == d[3:]:
                    temp.append((v[0], v[1]))
            simile_and_excelNumber_of_theSameVectors.append(temp)
        numberOfDistinctVectors = ('In Whole Dataset', count, len(numericalVS),
                                   str(round(count*100/len(numericalVS),1))+'%', simile_and_excelNumber_of_theSameVectors)


        ###################
        #print(numberOfDistinctVectors[2])
        #print(numberOfDistinctVectors[4])
        return distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectors





        # per distinct vector --> simile and number of vectors
    def groups_of_DistinctVector(self, simile_and_excelNumber_of_theSameVectors):
        s = Distinct_vectors
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
        #print(distinctVectors)
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


Distinct_vectors()