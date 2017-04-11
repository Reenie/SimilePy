import fileToMatrix as f
import main
import numpy as np

class VectorSpace_gender:
    def __init__(self):
        vsg = VectorSpace_gender
        numericalFeature_names, numericalVS = vsg.numericalVectorSpace_gender(vsg, main.filenames, numOfGender=2)
        file = main.dataset_path + "numericalVectorSpace_gender.txt"
        vsg.writeNumericalVSToFile(vsg, numericalVS, numericalFeature_names, file)



    def writeNumericalVSToFile(self, numericalVS, numericalFeature_names, file):
        file = open(file, "w")
        for col in numericalFeature_names:
            file.write(str(col) + " # ")
        file.write("\n")
        for row in numericalVS:
            for col in row:
                file.write(str(col) + " # ")
            file.write("\n")
        file.close()


    #def numericalVectorSpace_3genders_(self, filenames)

    #if numOfGender = 2, M and F is the first gender and N is the second.
    def numericalVectorSpace_gender(self, filenames, numOfGender=3):
        categoricalVS, categoricalFeature_names = f.FileToMatrix.VectorSpace_InWoleDataset_fromSecondColumn_usefulAttr(filenames)
        numericalFeature_names = VectorSpace_gender.numFeatureNames(VectorSpace_gender, categoricalVS, categoricalFeature_names)
        #print(numericalFeature_names)
        numericalVS = []
        len_ = len(numericalFeature_names)
        for row in categoricalVS:
            col_index = -1
            numerical_row = [0]*len_  #array initialization to zero
            for c in row:
                col_index +=1
                if categoricalFeature_names[col_index] == "Simile_id":
                        index = numericalFeature_names.index("Simile_id")
                        numerical_row[index]=c
                elif categoricalFeature_names[col_index] == "SIM":
                    if c == "1":
                        index = numericalFeature_names.index("SIM")
                        numerical_row[index]=1
                elif categoricalFeature_names[col_index] == "GENDER":
                    if c == "M":
                        index = numericalFeature_names.index("GENDER")
                        numerical_row[index] = 1
                    elif c == "F":
                        index = numericalFeature_names.index("GENDER")
                        if (numOfGender == 2):
                            numerical_row[index] = 1
                        else:
                            numerical_row[index] = 2
                    elif c == "N":
                        index = numericalFeature_names.index("GENDER")
                        if (numOfGender == 2):
                            numerical_row[index] = 2
                        else:
                            numerical_row[index] = 3
                elif categoricalFeature_names[col_index] == "DETERMINER":
                    if c == "det":
                        index = numericalFeature_names.index("DETERMINER")
                        numerical_row[index] = 1
                elif categoricalFeature_names[col_index] == "LEMMA":
                    if c.replace(" ", "")!="normal":
                        numerical_feature_name = "L_" + c.replace(" ", "")
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
                elif categoricalFeature_names[col_index] == "TEN_GEN_SEMS":
                    if c.replace(" ", "")!="normal":
                        numerical_feature_name = "T_" + c.replace(" ", "")
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
                else:
                    if c.replace(" ", "")!="normal":
                        numerical_feature_name = c.replace(" ", "")
                        index = numericalFeature_names.index(numerical_feature_name)
                        numerical_row[index] = 1
            numericalVS.append(numerical_row)
        #np.array(numericalFeature_names)
        numericalVS = np.array(numericalVS)
        temp = numericalFeature_names[2]
        numericalFeature_names[2] = numericalFeature_names[1]
        numericalFeature_names[1] = temp
        temp1 = np.copy(numericalVS[:, 2])
        numericalVS[:, 2] = numericalVS[:, 1]
        numericalVS[:, 1] = temp1
        #print(numericalFeature_names)
        #print("\n")
        #print(numericalVS)
        return numericalFeature_names, numericalVS



    # it returns an array with numerical feature names
    def numFeatureNames(self, categoricalVS, categoricalFeature_names):
        numerical_feature_names = []
        # index_numFeatures = -1
        index_catFeatures = -1
        for c in categoricalFeature_names:
            index_catFeatures += 1
            if index_catFeatures == 0:
                numerical_feature_names.append('Simile_id')
            elif index_catFeatures == 1:
                numerical_feature_names.append('SIM')
            elif index_catFeatures == 2:
                numerical_feature_names.append('GENDER')
            elif index_catFeatures == 6:
                numerical_feature_names.append('DETERMINER')
            else:
                numerical_features = self.numericalFeatures('self', categoricalVS, index_catFeatures, c)
                #print(numerical_features)
                for n in numerical_features:
                    numerical_feature_names.append(n)
        #print(numerical_feature_names)
        return numerical_feature_names


    # it returns array with numerical feature names of a specific categorical feature
    def numericalFeatures(self, categoricalVS, indexOfFeature, categorical_feature_name):
        vs = self  # this class
        numerical_feature_names = []
        for row in categoricalVS:
            row_val = row[indexOfFeature].replace(" ", "")
            if row_val != "normal":
                val = ""
                if categorical_feature_name == "LEMMA":
                    val = "L_" + row_val
                elif categorical_feature_name == "TEN_GEN_SEMS":
                    val = "T_" + row_val
                else:
                    val = row_val
                if not (val in numerical_feature_names):
                    numerical_feature_names.append(val)
        return numerical_feature_names




VectorSpace_gender()
