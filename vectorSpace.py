import fileToMatrix
import main


class vectorSpace:

    def createVectorSpace(self):


    def newFeatureName(self):
        categoricalVS, feature_names = fileToMatrix.VectorSpace_InWoleDataset_fromSecondColumn_usefulAttr(main.filenames)
        new_features = []
        count_row = 0
        for row in categoricalVS:
            count_row+=1
            count_col = 0
            for column in row:
                count_col+=1


    #
    def featureHashTable(matrix, indexOfFeature):
        vs = vectorSpace
        hashtable = {}  # key - frequent
        numOfKeys = 0
        for row in matrix:
            val = row[indexOfFeature].split(",")
            for v in val:
                if vs.hasKey(hashtable, v.replace(" ", "")):
                    if v.replace(" ", "") != "null":
                        hashtable[v.replace(" ", "")] += 1
                else:
                    if v.replace(" ", "") != "null":
                        hashtable[v.replace(" ", "")] = 1
                        numOfKeys += 1
        return numOfKeys, hashtable

    #
    def hashTableKeys(hashTable):
        keys = []
        for k in hashTable.keys():
            keys.append(k)
        return keys

    def hasKey(hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0