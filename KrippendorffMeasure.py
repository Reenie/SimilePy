import krippendorff
import numpy as np
import main as main
import xlrd

class KrippendorffMeasure:

    def __init__(self):
        s = KrippendorffMeasure
        s.printKrippendorffAndFleiss(self)
        #s.krippendorffAlpha(self, listOfXLSXFiles=s.files)
        #s.fleissKappa(self, listOfXLSXFiles = s.files)
        #s.krippendorfMeasure_test(self, s.matk)
        #s.computeFleissKappa(self, s.matf)

    files_MWEagreement = [
        "8_malakos_san_voutiro_MWEagreement.xlsx",
        "9_geros_san_tavros_MWEagreement.xlsx",
        "11_kokkinos san paparouna_MWEagreement.xlsx",
        "12_ntimenos_san_astakos_MWEagreement.xlsx",
        "18_Mavros_san_skotadi_MWEagreement.xlsx",
        "19_mperdemenos_san_to_koubari_MWEagreement.xlsx"
    ]

    files_semsAnnot = [
        "1_aspros_san_to_pani.2_semannot.xlsx",
        "18_mavros_san_skotadi.2_semannot.xlsx",
        "19_mperdemenos_san_to_kouvari.2_semannot.xlsx",
        "20_aspros_san_to_xioni.2_semannot.xlsx"
    ]

    files_freeAdjAnnot = [
        "apalos_comparative_interAnnot.xlsx",
        "aspros_comparative_interAnnot.xlsx",
        "geros_comparative_interAnnot.xlsx",
        "kokkinos_comparative_interAnnot.xlsx"
    ]


    def printKrippendorffAndFleiss(self):
        s = KrippendorffMeasure
        #MWE agreement
        a_list = s.krippendorffAlpha(self, listOfXLSXFiles=s.files_MWEagreement, mode = "mweAgreement")
        k_list = s.fleissKappa(self, listOfXLSXFiles=s.files_MWEagreement, mode = "mweAgreement")
        print("MWE Type Inter-Annotator Agreement")
        print('%-34s%-24s%-30s' % ("File", "Krippendorff's Alpha", "Fleiss' Kappa"))
        for a, k in zip(a_list, k_list):
            print('%-35s%-25s%-30s' % (a[0].replace("_MWEagreement.xlsx", ""), round(a[1],4), round(k[1], 4)))

        #Semantics agreement
        a_list = s.krippendorffAlpha(self, listOfXLSXFiles=s.files_semsAnnot, mode="semAgreement")
        k_list = s.fleissKappa(self, listOfXLSXFiles=s.files_semsAnnot, mode="semAgreement")
        print("\nSemantics Inter-Annotator Agreement")
        print('%-34s%-24s%-30s' % ("File", "Krippendorff's Alpha", "Fleiss' Kappa"))
        for a, k in zip(a_list, k_list):
            print('%-35s%-25s%-30s' % (a[0].replace("_semannot.xlsx", ""), round(a[1], 4), round(k[1], 4)))


        # free Adj agreement
        a_list = s.krippendorffAlpha(self, listOfXLSXFiles=s.files_freeAdjAnnot, mode="freeAdjAgreement")
        k_list = s.fleissKappa(self, listOfXLSXFiles=s.files_freeAdjAnnot, mode="freeAdjAgreement")
        print("\nFree Adjectives Inter-Annotator Agreement")
        print('%-34s%-24s%-30s' % ("File", "Krippendorff's Alpha", "Fleiss' Kappa"))
        for a, k in zip(a_list, k_list):
            print('%-35s%-25s%-30s' % (a[0].replace("_comparative_interAnnot.xlsx", ""), round(a[1], 4), round(k[1], 4)))


    def fleissKappa(self, listOfXLSXFiles = [], mode = "mweAgreement"):
        s = KrippendorffMeasure
        annotator_values_inWholeDataset = []
        k_list = []
        for file in listOfXLSXFiles:
            if mode == "mweAgreement":
                annotator_values = s.readXLSXFile_mweAgreement(self,  xlsx_filename = file)
            elif mode == "semAgreement":
                annotator_values = s.readXLSXFile_semAgreement(self, xlsx_filename=file)
            elif mode == "freeAdjAgreement":
                annotator_values = s.readXLSXFile_freeAdjAgreement(self, xlsx_filename=file)
            else:
                print("Error: unknown mode")
            reliability_matrix = s.fleissReliabilityMatrix(self, annotator_values)
            k = s.computeFleissKappa(self, reliability_matrix, DEBUG = False)
            k_list.append((file, k))
            annotator_values_inWholeDataset = annotator_values_inWholeDataset + annotator_values
        reliability_matrix_inWholeDataset = s.fleissReliabilityMatrix(self, annotator_values_inWholeDataset)
        k = s.computeFleissKappa(self, reliability_matrix_inWholeDataset, DEBUG=False)
        k_list.append(("All", k))
        #print("fleiss kappa = " + str(k_list))
        return k_list



    def krippendorffAlpha(self, listOfXLSXFiles = [], mode = "mweAgreement"):
        s = KrippendorffMeasure
        annotator_values_inWholeDataset = []
        k_list = []
        for file in listOfXLSXFiles:
            if mode == "mweAgreement":
                annotator_values = s.readXLSXFile_mweAgreement(self, xlsx_filename=file)
            elif mode == "semAgreement":
                annotator_values = s.readXLSXFile_semAgreement(self, xlsx_filename=file)
            elif mode == "freeAdjAgreement":
                annotator_values = s.readXLSXFile_freeAdjAgreement(self, xlsx_filename=file)
            else:
                print("Error: unknown mode")
            reliability_matrix = s.krippendorffReliabilityMatrix(self, annotator_values)
            #levellevel_of_measurement='interval'
            levellevel_of_measurement = 'nominal'
            a = krippendorff.alpha(reliability_data=reliability_matrix, value_counts=None, level_of_measurement=levellevel_of_measurement)
            k_list.append((file, a))
            annotator_values_inWholeDataset = annotator_values_inWholeDataset + annotator_values
        reliability_matrix_inWholeDataset = s.krippendorffReliabilityMatrix(self, annotator_values_inWholeDataset)
        #print(reliability_matrix_inWholeDataset)
        a = krippendorff.alpha(reliability_data=reliability_matrix_inWholeDataset, value_counts=None, level_of_measurement='interval')
        k_list.append(("All", a))
        #print("a = " + str(k_list))
        return k_list

    def readXLSXFile_freeAdjAgreement(self, xlsx_filename=""):
        s = KrippendorffMeasure  # self class
        xlsxFile = main.dataset_path + "inter_annotator_agreement/freeAdjectives_interannotator/" + xlsx_filename
        xlsx = xlrd.open_workbook(xlsxFile, encoding_override='utf-8')
        sheet = xlsx.sheet_by_index(0)  # get a sheet
        numberOfXlsxRows = sheet.nrows
        annotator_values = []
        for rowNum in range(numberOfXlsxRows):
            # xlsxRow_index = rowNum + 1
            if (rowNum < 201):
                cellValue_1 = str(sheet.cell(rowNum, 3).value).strip()
                cellValue_2 = str(sheet.cell(rowNum, 4).value).strip()
                cellValue_3 = str(sheet.cell(rowNum, 5).value).strip()
                if (cellValue_1 != "" and cellValue_2 != "" and cellValue_3 != ""):
                    annotator_values.append([cellValue_1, cellValue_2, cellValue_3])
        # print(annotator_values, end="\n\n")
        return annotator_values



    def readXLSXFile_semAgreement(self, xlsx_filename=""):
        s = KrippendorffMeasure  # self class
        xlsxFile = main.dataset_path + "inter_annotator_agreement/simile_semantics_interannotator/" + xlsx_filename
        xlsx = xlrd.open_workbook(xlsxFile, encoding_override='utf-8')
        sheet = xlsx.sheet_by_index(0)  # get a sheet
        numberOfXlsxRows = sheet.nrows
        annotator_values = []
        for rowNum in range(numberOfXlsxRows):
            # xlsxRow_index = rowNum + 1
            # if (xlsxRow_index < maxXlsxRows + 1):
            cellValue = str(sheet.cell(rowNum, 1).value).strip()
            # print(cellValue)
            if cellValue == "1.0" or cellValue == "1":
                cellValue_1 = str(sheet.cell(rowNum, 6).value).strip()
                cellValue_2 = str(sheet.cell(rowNum, 7).value).strip()
                cellValue_3 = str(sheet.cell(rowNum, 8).value).strip()
                if (cellValue_1 != "" and cellValue_2 != "" and cellValue_3 != ""):
                    annotator_values.append([cellValue_1, cellValue_2, cellValue_3])
        # print(annotator_values, end="\n\n")
        return annotator_values

    def readXLSXFile_mweAgreement(self,  xlsx_filename = ""):
        s = KrippendorffMeasure # self class
        xlsxFile = main.dataset_path + "inter_annotator_agreement/" + xlsx_filename
        xlsx = xlrd.open_workbook(xlsxFile, encoding_override='utf-8')
        sheet = xlsx.sheet_by_index(0)  # get a sheet
        numberOfXlsxRows = sheet.nrows
        annotator_values = []
        for rowNum in range(numberOfXlsxRows):
            #xlsxRow_index = rowNum + 1
            #if (xlsxRow_index < maxXlsxRows + 1):
            cellValue = str(sheet.cell(rowNum, 1).value).strip()
            #print(cellValue)
            if cellValue == "1.0" or cellValue == "1" :
                cellValue_1 = str(sheet.cell(rowNum, 7).value).strip()
                cellValue_2 = str(sheet.cell(rowNum+1, 7).value).strip()
                cellValue_3 = str(sheet.cell(rowNum+2, 7).value).strip()
                if (cellValue_1 != "" and cellValue_2 != "" and cellValue_3 != ""):
                    annotator_values.append([cellValue_1, cellValue_2, cellValue_3])
        #print(annotator_values, end="\n\n")
        return annotator_values



    def categoriesVector(self, annotator_values):
        s = KrippendorffMeasure
        categories = []
        for vv in annotator_values:
              for v in vv:
                  if s.findValueInVector(self, v, categories)==-1:
                      categories.append(v)
        ############3
        #print(categories)
        return categories

    def fleissReliabilityMatrix(self, annotator_values):
        s = KrippendorffMeasure
        reliability_matrix = []
        numberoOfInstances = len(annotator_values)
        categories = s.categoriesVector(self, annotator_values)
        numOfcategories = len(categories)
        for i in range(numberoOfInstances):
            temp = []
            for i in range(numOfcategories):
                temp.append(0)
            reliability_matrix.append(temp)
        #############3
        #print(reliability_matrix)
        index = -1
        for vv in annotator_values:
            index += 1
            for v in vv:
                pos = s.findValueInVector(self, v, categories)
                reliability_matrix[index][pos] += 1
        ###################
        #print(reliability_matrix)
        return reliability_matrix


    def krippendorffReliabilityMatrix(self, annotator_values):
        s = KrippendorffMeasure
        reliability_matrix = []
        numberoOfAnnotators = len(annotator_values[0])
        categories = s.categoriesVector(self, annotator_values)
        numOfUnits = len(annotator_values)
        for i in range(numberoOfAnnotators):
            temp = []
            for i in range(numOfUnits):
                temp.append(0)
            reliability_matrix.append(temp)
        #############3
        #print(reliability_matrix)
        unit_index = -1
        for vv in annotator_values:
            unit_index += 1
            annotator_index = -1
            for v in vv:
                annotator_index +=1
                pos = s.findValueInVector(self, v, categories)
                reliability_matrix[annotator_index][unit_index] = pos+1
        ###################
        #print(reliability_matrix)
        return reliability_matrix


    def krippendorffReliabilityMatrix_old(self, annotator_values):
        s = KrippendorffMeasure
        reliability_matrix = []
        numberoOfAnnotators = len(annotator_values[0])
        categories = s.categoriesVector(self, annotator_values)
        numOfcategories = len(categories)
        for i in range(numberoOfAnnotators):
            temp = []
            for i in range(numOfcategories):
                temp.append(0)
            reliability_matrix.append(temp)
        #############3
        #print(reliability_matrix)
        for vv in annotator_values:
            annotator_index = -1
            for v in vv:
                annotator_index +=1
                pos = s.findValueInVector(self, v, categories)
                reliability_matrix[annotator_index][pos] += 1
        ###################
        #print(reliability_matrix)
        return reliability_matrix

    ###if the value exists in vector then it returns the index of the value, otherwise it returns -1
    def findValueInVector(self, value, vector):
        index = -1
        for v in vector:
            index += 1
            if value == v:
                return index
        return -1



    matf = [
        [4, 0, 0],
        [1, 3, 0],
        [3, 1, 0],
        [0, 0, 4],
        [0, 1, 3]
    ]



    matk = [
        [2, 1, 2],
        [2, 1, 2],
        [2, 1, 2],
        [2, 1, 2]
    ]

    matkk = [
        [0, 4],
        [3, 1],
        [1, 3]
    ]

    mat = [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7]
        ]

    def krippendorfMeasure_test(self, vector):
        vector = np.array(vector)
        a = krippendorff.alpha(reliability_data = vector, value_counts=None, level_of_measurement='nominal')
        #print("a = " + str(a))


    #Computes the Fleiss' Kappa value as described in (Fleiss, 1971) """
    def computeFleissKappa(self, mat, DEBUG=True):
        s = KrippendorffMeasure

        def checkEachLineCount(mat):
            """ Assert that each line has a constant number of ratings
                @param mat The matrix checked
                @return The number of ratings
                @throws AssertionError If lines contain different number of ratings """
            n = sum(mat[0])
            assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
            return n

        """ Computes the Kappa value
            @param n Number of rating per subjects (number of human raters)
            @param mat Matrix[subjects][categories]
            @return The Kappa value """
        n = checkEachLineCount(mat) # PRE : every line count must be equal to n
        N = len(mat)
        k = len(mat[0])
        if DEBUG:
            print(n, "raters.")
            print(N, "subjects.")
            print(k, "categories.")
        # Computing p[]
        p = [0.0] * k
        for j in range(k):
            p[j] = 0.0
            for i in range(N):
                p[j] += mat[i][j]
            p[j] /= N * n
        if DEBUG:
            print("p =", p)
        # Computing P[]
        P = [0.0] * N
        for i in range(N):
            P[i] = 0.0
            for j in range(k):
                P[i] += mat[i][j] * mat[i][j]
            P[i] = (P[i] - n) / (n * (n - 1))
        if DEBUG:
            print("P =", P)
        # Computing Pbar
        Pbar = sum(P) / N
        if DEBUG:
            print("Pbar =", Pbar)
        # Computing PbarE
        PbarE = 0.0
        for pj in p:
            PbarE += pj * pj
        if DEBUG:
            print("PbarE =", PbarE)
        kappa = (Pbar - PbarE) / (1 - PbarE)
        if DEBUG:
            print("kappa =", kappa)
        return kappa






KrippendorffMeasure()