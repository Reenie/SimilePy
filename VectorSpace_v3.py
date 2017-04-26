import main


class VectorSpace_v3:
    attrForVectorSpace = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #all atributes
    attrWithMultipleCategoricalValues = [3, 4, 5, 12] #numerical vector space has one feature for each categorical value
    attrWithNumericalValues = [0, 1, 2]  #numerical feature has the same value as the catigorical one

          #[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    full_attr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #it should be changed
    some_attr = [0, 1, 2, 3,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #it should be changed


    attrForVectorSpace = some_attr #it should be changed

    txtHeaders = ["SIMILE", #0
                  "XLSX_ROW", #1
                  "TXT_ROW",  # 2
                  "GENDER",  # 3
                  "LEMMA", #4
                  "TEN_GEN_SEMS",  # 5  TENOR SEM GENERALISATION
                  "MWE_TYPE",  # 6
                  "DETERMINER",  # 7
                  "EMPM",  # 8
                  "EMPP",  # 9
                  "IWO",  # 10
                  "IXP-CREATIVE",  #11
                  "IXP-N_W_PUNC",  # 12
                  "MOD",  # 13
                  "AGR",  # 14
                  "MWO",  # 15
                  "VAR"]  # 16



    txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = VectorSpace_v3
        #s.featureMatrix(self, main.filenames, attrForVecotrSpace=s.attrForVectorSpace)
        s.numericalVectorSpace(self, main.filenames)


    # VectorSpace in whole dataset with usefull attributes only
    def featureMatrix(self, filenames, attrForVecotrSpace = []):
        s = VectorSpace_v3
        matrix = []
        feature_names = []
        flag = 0
        for file in filenames:
            with open(s.txt_datapath + "" + file, encoding="utf8") as f:
                if flag == 0:
                    for c in s.attrForVectorSpace:
                        feature_names.append(s.txtHeaders[c])
                    flag = 1  # greater than 1
                firstLine_flag = 0
                for line in f:
                    if firstLine_flag == 0:  #discard first line
                        firstLine_flag = 1
                    else:
                        splited_row = (line.strip()).split("#")
                        rowVector = []
                        for c in s.attrForVectorSpace:
                            rowVector.append(splited_row[c].strip())
                        matrix.append(rowVector)
        #print(feature_names)
        #print(matrix)
        return matrix, feature_names

    def numericalVectorSpace(self, filenames):
        s = VectorSpace_v3 #This class
        categoricalVS, categoricalFeature_names = s.featureMatrix("self", filenames, attrForVecotrSpace=s.attrForVectorSpace)
        numericalFeature_names = s.numFeatureNames("self", categoricalVS, categoricalFeature_names, s.attrForVectorSpace)
        # print(numericalFeature_names)
        numericalVS = []
        len_ = len(numericalFeature_names)
        for row in categoricalVS:
            col_index = -1
            numerical_row = [0] * len_  # array initialization to zero
            for c, a in zip(row, s.attrForVectorSpace):
                col_index += 1
                if a not in s.attrWithMultipleCategoricalValues:
                    if a in s.attrWithNumericalValues:  #these features take the same values as categorical ones
                        index = numericalFeature_names.index(s.txtHeaders[a])
                        numerical_row[index] = c
                    else:
                        if c.replace(" ", "") != "normal" and c.replace(" ", "") != "nodet":
                            index = numericalFeature_names.index(s.txtHeaders[a])
                            numerical_row[index] = 1
                elif categoricalFeature_names[col_index] == "LEMMA":
                    if c.replace(" ", "") != "normal":
                        numerical_feature_name = "L_" + c.strip() #replace(" ", "")
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
                elif categoricalFeature_names[col_index] == "TEN_GEN_SEMS":
                    if c.replace(" ", "") != "normal":
                        numerical_feature_name = "T_" + c.strip() #replace(" ", "")
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
                else:
                    if c.replace(" ", "") != "normal":
                        numerical_feature_name = c.replace(" ", "")
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
            numericalVS.append(numerical_row)
        #print(numericalFeature_names)
        #print("\n")
        #print(numericalVS)
        #print("Number of features: " +str(len(numericalFeature_names)))
        return numericalFeature_names, numericalVS



        # it returns an array with numerical feature names

    def numFeatureNames(self, categoricalVS, categoricalFeature_names, attrForVectorSpace):
        s = VectorSpace_v3 #this class
        numerical_feature_names = []
        # index_numFeatures = -1
        index_catFeatures = -1
        for c, a in zip(categoricalFeature_names, attrForVectorSpace):
            index_catFeatures += 1
            if a not in s.attrWithMultipleCategoricalValues:
                numerical_feature_names.append(c)
            else:
                numerical_features = s.numericalValuesOfFeature("self", categoricalVS, index_catFeatures, c)
                # print(numerical_features)
                for n in numerical_features:
                    numerical_feature_names.append(n)
        #print(numerical_feature_names)
        return numerical_feature_names


        # it returns array with numerical feature names of a specific categorical feature

    def numericalValuesOfFeature(self, categoricalVS, indexOfFeature, categorical_feature_name):
        s = VectorSpace_v3  # this class
        values = []
        for row in categoricalVS:
            row_val = row[indexOfFeature].strip() #replace(" ", "")
            if row_val != "normal":
                val = ""
                if categorical_feature_name == "LEMMA":
                    val = "L_" + row_val
                elif categorical_feature_name == "TEN_GEN_SEMS":
                    val = "T_" + row_val
                else:
                    val = row_val
                if not (val in values):
                    values.append(val)
        return values

    #matrix transposition
    def rowToColTransposition(self, m):
        rows = len(m)  # rows
        cols = len(m[0])  # columns
        mT = []
        for c in range(0, cols):
            mT.append([])
            for r in range(0, rows):
                mT[c].append(m[r][c])
        return mT



VectorSpace_v3()
