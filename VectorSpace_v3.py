import main


class VectorSpace_v3:
    attrForVectorSpace = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    txtHeaders = ["SIMILE", #0
                  "XLSX ROW", #1
                  "TXT ROW",  # 2
                  "GENDER",  # 3
                  "LEMMA", #4
                  "TENOR SEM",  # 5  TENOR SEM GENERALISATION
                  "MWE TYPE",  # 6
                  "DETERMINER",  # 7
                  "EMPM",  # 8
                  "EMPP",  # 9
                  "IWO",  # 10
                  "IXP-CREATIVE",  #11
                  "IXP-N, IXP-W, IXP-PUNC",  # 12
                  "MOD",  # 13
                  "AGR",  # 14
                  "MWO",  # 15
                  "VAR"]  # 16

    txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = VectorSpace_v3
        s.featureMatrix(self, main.filenames, attrForVecotrSpace=s.attrForVectorSpace)


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
        print(feature_names)
        print(matrix)
        return matrix, feature_names


VectorSpace_v3()
