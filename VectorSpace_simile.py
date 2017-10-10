import main


class VectorSpace_simile:
    full_attr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] #all atributes
    attrWithMultipleCategoricalValues = [3, 4, 5, 6, 7] #numerical vector space has one feature for each categorical value
    attrWithNumericalValues = [0, 1, 2]  #numerical feature has the same value as the catigorical one

          #[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    full_attr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] #it should be changed
    some_attr = [3, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]#it should be changed
    numOfGenders = 2 #it should be changed to 3 or 2

    attrForVectorSpace = some_attr #it should be changed



    txtHeaders = ["",  #FILE 0
               "TEXT",  # 1
               "SIMILE",  # 2
               "GENDER",  # 3
               "HEAD",  # 4
               "LEMMA",  # 5
               "MOD_PRED_SEMS",  # 6   MODIFIED PRED SEMS
               "TEN_GEN_SEMS",  # 7  TENOR SEM GENERALISATION
               "MWE_TYPE",  # 8
               "PHENOMENON",  # 9
               "DETERMINER",  # 10
               "EMPM",  # 11
               "EMPP",  # 12
               "COMP",  # 13
               "IWO",  # 14
               "IXP-CREATIVE",  # 15
               "IXP-EXPANSION",  # 16
               "IXP-N",  # 17
               "IXP-W",  # 18
               "IXP-PUNC",  # 19
               "MOD",  # 20
               "AGR",  # 21
               "MWO",  # 22
               "VAR",  # 23
               "AN"]  # 24



    txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = VectorSpace_simile
        #s.featureMatrix(self, main.filenames, attrForVecotrSpace=s.attrForVectorSpace)
        s.numericalVectorSpace(self, main.filenames, gender=s.numOfGenders )


    # VectorSpace in whole dataset with usefull attributes only
    def featureMatrix(self, filenames, attrForVectorSpace = []):
        s = VectorSpace_simile
        matrix = []
        feature_names = []
        flag = 0
        for file in filenames:
            with open(s.txt_datapath + "" + file, encoding="utf8") as f:
                if flag == 0:  #first line
                    for c in attrForVectorSpace:
                        feature_names.append(s.txtHeaders[c])
                    flag = 1  # greater than 1
                    ###########
                    print(feature_names)
                if flag == 1:
                    for line in f:
                        splited_row = (line.strip()).split("#")
                        #########
                        #print(splited_row)
                        rowVector = []
                        for c in attrForVectorSpace:
                            rowVector.append(splited_row[c].strip())
                            ###########
                            #print(c)
                            #print(rowVector)
                        matrix.append(rowVector)
        #print(feature_names)
        #print(matrix)
        return matrix, feature_names

    def numericalVectorSpace(self, filenames, gender=numOfGenders):
        s = VectorSpace_simile #This class
        categoricalVS, categoricalFeature_names = s.featureMatrix("self", filenames, attrForVectorSpace=s.attrForVectorSpace)
        numericalFeature_names = s.numFeatureNames("self", categoricalVS, categoricalFeature_names, s.attrForVectorSpace, gender=gender)
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
                elif categoricalFeature_names[col_index] == "GENDER" and gender==2:
                    if c.replace(" ", "") == "N":
                        numerical_feature_name = "N"
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
                    if c.replace(" ", "") == "M" or c.replace(" ", "") == "F":
                        numerical_feature_name = "M/F"
                        index = numericalFeature_names.index(numerical_feature_name)
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

    def numFeatureNames(self, categoricalVS, categoricalFeature_names, attrForVectorSpace, gender=3):
        s = VectorSpace_simile #this class
        numerical_feature_names = []
        # index_numFeatures = -1
        index_catFeatures = -1
        for c, a in zip(categoricalFeature_names, attrForVectorSpace):
            index_catFeatures += 1
            if a not in s.attrWithMultipleCategoricalValues:
                numerical_feature_names.append(c)
            else:
                numerical_features = s.numericalValuesOfFeature("self", categoricalVS, index_catFeatures, c, gender=gender)
                # print(numerical_features)
                for n in numerical_features:
                    numerical_feature_names.append(n)
        #print(numerical_feature_names)
        return numerical_feature_names


        # it returns array with numerical feature names of a specific categorical feature

    def numericalValuesOfFeature(self, categoricalVS, indexOfFeature, categorical_feature_name, gender=3):
        s = VectorSpace_simile  # this class
        values = []
        for row in categoricalVS:
            row_val = row[indexOfFeature].strip() #replace(" ", "")
            if row_val != "normal":
                val = ""
                if categorical_feature_name == "LEMMA":
                    val = "L_" + row_val
                elif categorical_feature_name == "TEN_GEN_SEMS":
                    val = "T_" + row_val
                elif categorical_feature_name == "GENDER" and gender==2:
                    if row_val == "N":
                        val = "N"
                    elif row_val=="M" or row_val=="F":
                        val = "M/F"
                    else:
                        print("unknown gender")
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



VectorSpace_simile()
