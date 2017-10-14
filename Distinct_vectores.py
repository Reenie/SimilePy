import main as main
import VectorSpace_simile as VSsimile

class Distinct_vectors:

    #txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = Distinct_vectors
        #s.distinctNumericalVectorSpace(self)
        s.printStatistics_of_DistinctVectors(self)

    def printStatistics_of_DistinctVectors(self):
        s = Distinct_vectors
        distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectorsPerSimile = s.distinctNumericalVectorSpace(self)
        a, b, n = s.distinctNumericalVectorSpace_inWholeDataset(self)
        print("\r\n\r\n")
        print('%-35s%-20s%-15s%-15s' % ("File Name", "Distinct Vectors", "All Vectors", "Percentage"))
        for e in numberOfDistinctVectorsPerSimile:
            print('\r%-35s%-20s%-15s%-15s' % (e[0], e[1], e[2], e[3]))
        print('\r%-35s%-20s%-15s%-15s' % (n[0], n[1], n[2], n[3]))
        print("\r\n\r\n")
        print("The excel cell numbers of the same feature vectors per simile:\r\n")
        for e in numberOfDistinctVectorsPerSimile:
            print(e[0])
            for ee in e[4]:
                print(ee)
            print("\r\n\r\n")



    def distinctNumericalVectorSpace(self):
        vss = VSsimile.VectorSpace_simile
        distinct_numericalVectorSpace = []
        numberOfDistinctVectorsPerSimile = []
        numericalFeature_names = []

        for filename in main.filenames:
            numericalFeature_names, numericalVS = vss.numericalVectorSpace(self, [filename], gender=vss.numOfGenders)
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
        numberOfDistinctVectors = ('In Whole Dataset', count, len(numericalVS), str(round(count*100/len(numericalVS),1))+'%')
        return distinct_numericalVectorSpace, numericalFeature_names, numberOfDistinctVectors



Distinct_vectors()