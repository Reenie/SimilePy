import scipy.stats
from scipy.stats.stats import pearsonr
from fileToMatrix import FileToMatrix
import sys
import main as main
import pandas
import texttable as tt


class chiSquaredTest:
    def __init__(self):
        chiSquaredTest.culcAndPrintChiSquereTest('self')




    """
    for i in range(1, 11):
        x.append([i, i ** 2, i ** 3])

    tab.add_rows(x)
    tab.set_cols_align(['r', 'r', 'r'])
    tab.header(['Number', 'Number Squared', 'Number Cubed'])
    print
    tab.draw()
    """

    def culcAndPrintChiSquereTest(self):
            results, features_list, p_table_allFiles = chiSquaredTest.calcChiSquereTest('self')
            print("CHI-SQUARED TEST\n")
            print("\r\nIf p-value > 0.1 the features are independent")
            table_allResults = []
            for r, pt in zip(results, p_table_allFiles):
                print("\r\n\r\n\r\n" + r[0]) #print filename
                for fr in r[1]: #for file_results in results
                    print("\r\n"+ fr[0] + " - " + fr[1])
                    print("\rList of Feature1 [value, occurrences, percentage] : " + str(fr[2]))
                    print("\rList of Feature2 [value, occurrences, percentage] : " + str(fr[3]))
                    msg = "\rX^2: {}\nDegrees of Freedom: {}\np-value: {}"
                    print(msg.format(fr[4], fr[6], fr[5])) #chi2, ddof, p
                table_allResults.append(["\r\n\r\n\r\n"+r[0], pandas.DataFrame(pt, features_list, features_list)])
            wholeDataset_results, featuresWholeDataset_list, p_table_wholeDataset = chiSquaredTest.calcChiSquereTest_wholeDataset('self')
            print("\r\n\r\n\r\nCHI-SQUARED TEST IN WHOLE DATASET")
            print("\r\nIf p-value > 0.1 the features are independent")
            for wdr in wholeDataset_results:  # for file_results in results
                print("\r\n" + wdr[0] + " - " + wdr[1])
                print("\rList of Feature1 [value, occurrences, percentage] : " + str(wdr[2]))
                print("\rList of Feature2 [value, occurrences, percentage] : " + str(wdr[3]))
                msg = "\rX^2: {}\nDegrees of Freedom: {}\np-value: {}"
                print(msg.format(wdr[4], wdr[6], wdr[5]))  # chi2, ddof, p
            table_allResults.append(["\r\n\r\n\r\nIN WHOLE DATASET\r\nIf p-value > 0.1 the features are independent", pandas.DataFrame(p_table_wholeDataset, featuresWholeDataset_list, featuresWholeDataset_list)])
            print("\r\n\r\n\r\nCHI-SQUARED TEST")
            print("\r\np-value in Tables")
            print("\r\nIf p-value > 0.1 the features are independent")
            for t in table_allResults:
                print(t[0])
                print(t[1])
            return


    # mode=1: simm with all, mode=2:gender with all
    def calcChiSquereTest(self):
        #features = ["Sim", "Gender", "Head", "Lemma", "ModPredSems", "MWEtype", "Phenomenon"]
        results = []
        p_table_allFiles = []
        cc = chiSquaredTest
        for file in main.filenames:
            results_of_file = []
            vectorSpace, features = FileToMatrix.fileToVectorSpace_from2ndColumn_usefulAttr(file) #we need only the list of features
            feature1_index = -1
            p_table_file = []
            complete_p_table_row_with_zero_index = 0
            for feature1 in features:
                feature1_index += 1
                feature2_index = -1
                p_table_row = []
                complete_p_table_row_with_zero_index += 1
                for x in range(0, complete_p_table_row_with_zero_index):
                    p_table_row.append(None)
                for feature2 in features:
                    feature1_hashtable = {}
                    feature2_hashtable = {}
                    feature2_index += 1
                    if (feature2_index > feature1_index):
                        try:
                            freq_table, feature_names_list, feature1_hashtable, feature2_hashtable = cc.freq_matrix(file, feature1_index, feature2_index)
                            chi2, p, ddof, expected = scipy.stats.chi2_contingency(freq_table)
                            results_of_file.append([feature1, feature2,
                                            cc.keyFreqPercentOfHashKeys(feature1_hashtable),
                                            cc.keyFreqPercentOfHashKeys(feature2_hashtable), chi2, p, ddof, expected])
                            p_table_row.append(p)
                        except Exception as e:
                            results_of_file.append([feature1, feature2,
                                                    cc.keyFreqPercentOfHashKeys(feature1_hashtable),
                                                    cc.keyFreqPercentOfHashKeys(feature2_hashtable), None, None, None,
                                                    []])
                            p_table_row.append(None)
                            print(str(e) + "\n")
                p_table_file.append(p_table_row)
            p_table_allFiles.append(p_table_file)
            results.append([str(file), results_of_file])
        return results, features, p_table_allFiles

        # returns freq_matrix, feature_names



    def freq_matrix(filename, indexOfFeature1, indexOfFeature2):
            cc = chiSquaredTest
            vectorSpace, feature_names_list = FileToMatrix.fileToVectorSpace_from2ndColumn_usefulAttr(filename)
            numOfFeature1Values, feature1_hashtable = cc.featureHashTable(vectorSpace, indexOfFeature1)
            numOfFeature2Values, feature2_hashtable = cc.featureHashTable(vectorSpace, indexOfFeature2)
            freq_table = []
            for i in range(0, numOfFeature1Values):  # for all values of feature1 add subtables
                a = []
                for j in range(0, numOfFeature2Values):  # subtables initialization
                    a.append(0)
                freq_table.append(a)
            for row in vectorSpace:
                index1 = -1
                for feature1 in feature1_hashtable.keys():
                    index1 += 1
                    row_splited1 = row[indexOfFeature1].split(",")
                    for rs1 in row_splited1:
                        if rs1.replace(" ", "") == feature1.replace(" ", ""):
                            index2 = -1
                            for feature2 in feature2_hashtable.keys():
                                index2 += 1
                                row_splited2 = row[indexOfFeature2].split(",")
                                for rs2 in row_splited2:
                                    if rs2.replace(" ", "") == feature2.replace(" ", ""):
                                        freq_table[index1][index2] += 1
            return freq_table, feature_names_list, feature1_hashtable, feature2_hashtable





    # mode=1: simm with all, mode=2:gender with all
    def calcChiSquereTest_wholeDataset(self):
        results = []
        p_table_wholeDataset = []
        cc = chiSquaredTest
        vectorSpace, features = FileToMatrix.VectorSpace_InWoleDataset_fromSecondColumn(
            main.filenames)  # we need only the list of features
        feature1_index = -1
        complete_p_table_row_with_zero_index = 0
        for feature1 in features:
            feature1_index += 1
            feature2_index = -1
            p_table_row = []
            complete_p_table_row_with_zero_index += 1
            for x in range(0, complete_p_table_row_with_zero_index):
                p_table_row.append(None)
            for feature2 in features:
                feature1_hashtable = {}
                feature2_hashtable = {}
                feature2_index += 1
                if (feature2_index > feature1_index):
                    try:
                        freq_table, feature_names_list, feature1_hashtable, feature2_hashtable = cc.freq_matrix_wholeDataset(
                            main.filenames, feature1_index, feature2_index)
                        chi2, p, ddof, expected = scipy.stats.chi2_contingency(freq_table)
                        results.append([feature1, feature2,
                                        cc.keyFreqPercentOfHashKeys(feature1_hashtable),
                                        cc.keyFreqPercentOfHashKeys(feature2_hashtable), chi2, p, ddof,
                                        expected])
                        p_table_row.append(p)
                    except Exception as e:
                        results.append([feature1, feature2,
                                        cc.keyFreqPercentOfHashKeys(feature1_hashtable),
                                        cc.keyFreqPercentOfHashKeys(feature2_hashtable), None, None, None,
                                            []])
                        p_table_row.append(p)
                        print(str(e) + "\n")
            p_table_wholeDataset.append(p_table_row)
        return results, features, p_table_wholeDataset




    def freq_matrix_wholeDataset(filenames, indexOfFeature1, indexOfFeature2):
        cc = chiSquaredTest
        vectorSpace, feature_names_list = FileToMatrix.VectorSpace_InWoleDataset_fromSecondColumn_usefulAttr(main.filenames)
        numOfFeature1Values, feature1_hashtable = cc.featureHashTable(vectorSpace, indexOfFeature1)
        numOfFeature2Values, feature2_hashtable = cc.featureHashTable(vectorSpace, indexOfFeature2)
        freq_table = []
        for i in range(0, numOfFeature1Values):  # for all values of feature1 add subtables
            a = []
            for j in range(0, numOfFeature2Values):  # subtables initialization
                a.append(0)
            freq_table.append(a)
        for row in vectorSpace:
            index1 = -1
            for feature1 in feature1_hashtable.keys():
                index1 += 1
                row_splited1 = row[indexOfFeature1].split(",")
                for rs1 in row_splited1:
                    if rs1.replace(" ", "") == feature1.replace(" ", ""):
                        index2 = -1
                        for feature2 in feature2_hashtable.keys():
                            index2 += 1
                            row_splited2 = row[indexOfFeature2].split(",")
                            for rs2 in row_splited2:
                                if rs2.replace(" ", "") == feature2.replace(" ", ""):
                                    freq_table[index1][index2] += 1
        # print("frequency table: " + str(freq_table))
        return freq_table, feature_names_list, feature1_hashtable, feature2_hashtable



    # It returns a list with frequent and percentage of hashtable keys
    def keyFreqPercentOfHashKeys(hashtable):
        cc = chiSquaredTest
        sum = cc.SumOfHashtableValues(hashtable)
        key_freq_percent = []
        for k in hashtable.keys():
            key_freq_percent.append([k, hashtable[k], str(round((hashtable[k] * 100) / sum, 2)) + "%"])
        return key_freq_percent

    # X^2 test for simile - gender
    def sim_gender_freq_matrix(filename):
        freqOfGender = [[0, 0, 0], [0, 0, 0]]  # frequences of [M, F, N]
        vectorSpace = FileToMatrix.fileToVectorSpace_fromSecondColumn(filename)
        for row in vectorSpace:
            if row[0].strip() == "0":
                if row[1].strip() == "M":
                    freqOfGender[0][0] += 1
                elif row[1].strip() == "F":
                    freqOfGender[0][1] += 1
                elif row[1].strip() == "N":
                    freqOfGender[0][2] += 1
                elif row[1].strip() == "M , F":
                    freqOfGender[0][0] += 1
                    freqOfGender[0][1] += 1
                else:
                    print("null gender:" + row[1].strip())
            elif row[0].strip() == "1":
                if row[1].strip() == "M":
                    freqOfGender[1][0] += 1
                elif row[1].strip() == "F":
                    freqOfGender[1][1] += 1
                elif row[1].strip() == "N":
                    freqOfGender[1][2] += 1
                elif row[1].strip() == "M , F":
                    freqOfGender[1][0] += 1
                    freqOfGender[1][1] += 1
                else:
                    print("null gender:" + row[1].strip())
            else:
                print("null simile:" + row[0].strip())
        print(freqOfGender)
        return freqOfGender


    #
    def featureHashTable(matrix, indexOfFeature):
        cc = chiSquaredTest
        hashtable = {}  # key - frequent
        numOfKeys = 0
        for row in matrix:
            val = row[indexOfFeature].split(",")
            for v in val:
                if cc.hasKey(hashtable, v.replace(" ", "")):
                    if v.replace(" ", "") != "null":
                        hashtable[v.replace(" ", "")] += 1
                else:
                    if v.replace(" ", "") != "null":
                        hashtable[v.replace(" ", "")] = 1
                        numOfKeys += 1
        return numOfKeys, hashtable

    #
    def countOfHashtableKeys(hashTable):
        count = 0
        for k in hashTable.keys():
            count += 1
        return count

    #
    def SumOfHashtableValues(hashTable):
        sum = 0
        for k in hashTable.keys():
            sum += hashTable[k]
        return sum

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

    # X^2 test for simile - any feature except gender
    # inexOfFeature --> sim: 0, lemma: 3, mwe: 4, phenomenon: 5
    def sim_feature_freq_matrix(filename, indexOfFeature):
        cc = chiSquaredTest
        vectorSpace = FileToMatrix.fileToVectorSpace_from2ndColumn_usefulAttr(filename)
        numOfFeatureValues, feature_hashtable = cc.featureHashTable(vectorSpace, indexOfFeature)
        a = []
        b = []
        for i in range(0, numOfFeatureValues):
            a.append(0)
            b.append(0)
        freq_table = [a, b]
        for row in vectorSpace:
            if row[0].strip() == "0":
                indexOfLemma = -1
                for lemma in feature_hashtable.keys():
                    indexOfLemma += 1
                    if row[indexOfFeature].strip() == lemma:
                        freq_table[0][indexOfLemma] += 1
                        break
            elif row[0].strip() == "1":
                indexOfLemma = -1
                for lemma in feature_hashtable.keys():
                    indexOfLemma += 1
                    if row[indexOfFeature].strip() == lemma:
                        freq_table[1][indexOfLemma] += 1
                        break
            else:
                print("null simile:" + row[0].strip())
        print(freq_table)
        return freq_table

    # X^2 test for simile - gender
    def gender_sim_freq_matrix(filename):
        freqOfGender = [[0, 0], [0, 0], [0, 0]]  # frequences of [M, F, N]
        vectorSpace = FileToMatrix.fileToVectorSpace_fromSecondColumn(filename)
        for row in vectorSpace:
            if row[1].strip() == "M":
                if row[0].strip() == "0":
                    freqOfGender[0][0] += 1
                elif row[0].strip() == "1":
                    freqOfGender[0][1] += 1
                else:
                    print("null simile:" + row[0].strip())
            elif row[1].strip() == "F":
                if row[0].strip() == "0":
                    freqOfGender[1][0] += 1
                elif row[0].strip() == "1":
                    freqOfGender[1][1] += 1
                else:
                    print("null simile:" + row[0].strip())
            elif row[1].strip() == "N":
                if row[0].strip() == "0":
                    freqOfGender[2][0] += 1
                elif row[0].strip() == "1":
                    freqOfGender[2][1] += 1
                else:
                    print("null simile:" + row[0].strip())
            elif row[1].strip() == "M , F":
                if row[0].strip() == "0":
                    freqOfGender[0][0] += 1
                    freqOfGender[1][0] += 1
                elif row[0].strip() == "1":
                    freqOfGender[0][1] += 1
                    freqOfGender[1][1] += 1
                else:
                    print("null simile:" + row[0].strip())
            else:
                print("null gender:" + row[1].strip())
        print(freqOfGender)
        return freqOfGender

    def cor(self):
        a = [1, 0, 1, 0]
        b = [1, 1, 0, 2]
        print(pearsonr(a, b))

    def chi_squere_test_example(self):
        fr = [[40, 5], [75, 60], [13, 7]]
        chi2, p, ddof, expected = scipy.stats.chi2_contingency(fr)
        msg = "Test Statistic: {}\np-value: {}\nDegrees of Freedom: {}\n"
        print(msg.format(chi2, p, ddof))
        print(expected)


chiSquaredTest()

