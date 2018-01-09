import main
from Old import fileToMatrix as f


class VectorSpace_v2:
    def __init__(self):
        numericalFeature_names, numericalVS = VectorSpace_v2.numericalVectorSpace('self', main.filenames)
        file = main.dataset_path + "numericalVectorSpace.txt"
        VectorSpace_v2.writeNumericalVSToFile(self, numericalVS, numericalFeature_names, file)



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


    def numericalVectorSpace(self, filenames):
        categoricalVS, categoricalFeature_names = f.FileToMatrix.VectorSpace_InWoleDataset_fromSecondColumn_usefulAttr(filenames)
        numericalFeature_names = VectorSpace_v2.numFeatureNames('self', categoricalVS, categoricalFeature_names)
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
        print(numericalFeature_names)
        print("\n")
        print(numericalVS)
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
            elif index_catFeatures == 6:
                numerical_feature_names.append('DETERMINER')
            else:
                numerical_features = VectorSpace_v2.numericalFeatures('self', categoricalVS, index_catFeatures, c)
                #print(numerical_features)
                for n in numerical_features:
                    numerical_feature_names.append(n)
        #print(numerical_feature_names)
        return numerical_feature_names


    # it returns array with numerical feature names of a specific categorical feature
    def numericalFeatures(self, categoricalVS, indexOfFeature, categorical_feature_name):
        vs = VectorSpace_v2  # this class
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




VectorSpace_v2()
