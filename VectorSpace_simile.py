import main


class VectorSpace_simile:
    full_attr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26] #all atributes
    attrWithMultipleCategoricalValues = [5, 6, 7, 8, 9, 10, 11, 12] #numerical vector space has one feature for each categorical value
    attrWithNumericalValues = [0, 1, 2]  #numerical feature has the same value as the catigorical one

    #[0, 1, 2, 5, 11, 15, 16, 18, 20, 22, 25]
    syntactic_attr = [0, 1, 2, 5, 11, 15, 16, 18, 20, 22, 25] #[0, 1, 2, 5, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26] #attributes without GENDER, MWE_TYPE and SEMANTICS
    semantic_attr = [0, 1, 2, 5, 9, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26]##attributes with GENDER, MWE_TYPE and SEMANTICS

    some_attr =[0, 1, 2, 13, 14, 17, 21, 23, 24, 26  ] #[0, 1, 2, 5, 9, 11,  13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26]#[0, 1, 2, 9]#it should be changed
    numOfGenders = 3 #it should be changed to 3 or 2

    attrForVectorSpace = some_attr #it should be changed



    txtHeaders = ["FILE",  #FILE 0
               "XLSX_ROW",  # 1
               "TXT_ROW",  #2
               "TEXT",  # 3
               "SIMILE",  # 4
               "GENDER",  # 5  YES
               "HEAD",  # 6
               "LEMMA",  # 7
               "MOD_PRED_SEMS",  # 8   MODIFIED PRED SEMS
                "SEMANTICS",  # 9 YES     GENERALIZED SEMANTICS
               "TEN_GEN_SEMS",  # 10  TENOR SEM GENERALISATION
               "MWE_TYPE",  # 11 YES
               "PHENOMENON",  # 12
               "DETERMINER",  # 13 YES
               "EMPM",  # 14 YES
               "EMPP",  # 15 YES
               "COMP",  # 16 YES
               "IWO",  # 17 YES
               "IXP-CREATIVE",  # 18 YES
               "IXP-EXPANSION",  # 19
               "IXP-N",  # 20 YES
               "IXP-W",  # 21 YES
               "IXP-PUNC",  # 22 YES
               "MOD",  # 23 YES
               "AGR",  # 24 YES
               "MWO",  # 25 YES
               "VAR"  # 26 YES
                  ]



    txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = VectorSpace_simile
        #s.featureMatrix(self, main.filenames, attrForVecotrSpace=s.attrForVectorSpace)
        s.numericalVectorSpace(self, main.filenames, gender=s.numOfGenders )


    # VectorSpace in whole dataset with usefull attributes only
    def featureMatrix(self, filenames, attrForVectorSpace = [], genders=numOfGenders):
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
                if flag == 1:
                    firstLine_flag=0
                    for line in f:
                        if(firstLine_flag == 0): #first line of each file
                            firstLine_flag = 1
                        else:
                            splited_row = (line.strip()).split("#")
                            rowVector = []
                            for c in attrForVectorSpace:
                                if(s.txtHeaders[c]=='GENDER' and genders == 2 ):
                                    #print(splited_row[0] + ", "+ splited_row[1] + ", " + splited_row[2] )
                                    if splited_row[c].strip() == "N":
                                        rowVector.append(splited_row[c].strip())
                                    elif splited_row[c].strip() == "M" or splited_row[c].strip() == "F" or splited_row[c].strip() == "M/F":
                                        rowVector.append('M/F')
                                else:
                                    rowVector.append(splited_row[c].strip())
                            matrix.append(rowVector)
        #print(feature_names)
        #print(matrix)
        #print("\n")
        return matrix, feature_names

    def numericalVectorSpace(self, filenames, gender=numOfGenders):
        s = VectorSpace_simile #This class
        ##################
        #print(s.attrForVectorSpace)
        categoricalVS, categoricalFeature_names = s.featureMatrix("self", filenames, attrForVectorSpace=s.attrForVectorSpace)
        ############3
        #print(categoricalFeature_names)
        #print(categoricalVS)
        numericalFeature_names = s.numFeatureNames("self", categoricalVS, categoricalFeature_names, s.attrForVectorSpace, gender=gender)
        #print(numericalFeature_names)
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
                    if c.replace(" ", "") == "M" or c.replace(" ", "") == "F" or c.replace(" ", "") == "M/F":
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
        #################
        #print(numericalFeature_names)
        #print("\n")
        #count = 0
        #numericalVS.sort(key=lambda x: (x[0], x[1]))
        #for i in numericalVS:
        #    print(i)
        #    count = count + 1
        #print(count)
        #print("Number of features: " +str(len(numericalFeature_names)))
        return numericalFeature_names, numericalVS



        # it returns an array with numerical feature names

    def numFeatureNames(self, categoricalVS, categoricalFeature_names, attrForVectorSpace, gender=numOfGenders):
        s = VectorSpace_simile #this class
        numerical_feature_names = []
        # index_numFeatures = -1
        index_catFeatures = -1
        ################
        #print(categoricalFeature_names)
        for c, a in zip(categoricalFeature_names, attrForVectorSpace):
            index_catFeatures += 1
            if a not in s.attrWithMultipleCategoricalValues:
                numerical_feature_names.append(c)
                ###########
                #print(categoricalFeature_names)
                #print(numerical_feature_names)
            else:
                numerical_features = s.numericalValuesOfFeature("self", categoricalVS, index_catFeatures, c, gender=gender)
                # print(numerical_features)
                for n in numerical_features:
                    numerical_feature_names.append(n)
        ####################
        #print(numerical_feature_names)
        return numerical_feature_names


        # it returns array with numerical feature names of a specific categorical feature

    def numericalValuesOfFeature(self, categoricalVS, indexOfFeature, categorical_feature_name, gender=numOfGenders):
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
                    #########
                    #print(row_val)
                    if row_val == "N":
                        val = "N"
                    elif row_val=="M" or row_val=="F" or row_val=="M/F":
                        val = "M/F"
                    else:
                        print("unknown gender: " + row_val + "- file: " + row[0]+ " row: " + row[1] + " textRow: " + row[2] )
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
