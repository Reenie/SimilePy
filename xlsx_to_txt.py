import xlrd
import timeit


class XlsxToTxt:
    Files = ["",
             "1_aspros_san_to_pani.1",
             "2_stolismenos_san_fregata.1",
             "3_apalos_san_poupoulo.1",
             "4_apalos_san_xadi.1",
             "5_elafrys_san_poupoulo.1",
             "6_kokkinos_san_astakos.1",
             "7_oplismenos_san_astakos.1",
             "8_malakos_san_voutiro.1",
             "9_geros_san_tavros.1",
             "10_pistos_san_skilos.1",
             "11_kokkinos_san_paparouna.1",
             "12_ntimenos_san_astakos.1",
             "13_kokkinos_san_to_pantzari.1",
             "14_glikos_san_meli.1",
             "15_aspros_san_to_gala.1",
             "16_krios_san_ton_pago.1",
             "17_grigoros_san_astrapi.1",
             "18_mavros_san_skotadi.1",
             "19_mperdemenos_san_to_kouvari.1",
             "20_aspros_san_to_xioni.1"
             ]

    linux_dataset_path = "/home/pkouris/Dropbox/EMP_DID_dropbox/NLP/SIMILES/dataset/"
    win_dataset_path = "E:/Dropbox/EMP_DID_dropbox/NLP/SIMILES/dataset/"
    dataset_path = win_dataset_path  # It should be changed according to pc
    txt_datapath = dataset_path + "txt/"

    cellsForTXT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    headers = ["",  # 0
               "TEXT",  # 1
               "SIMILE",  # 2
               "GENDER",  # 3
               "HEAD",  # 4
               "LEMMA",  # 5
               "MOD_PRED_SEMS",  # 6   MODIFIED PRED SEMS
               "GEN_SEMS",  #7     GENERALIZED SEMANTICS
               "TEN_GEN_SEMS",  # 8  TENOR SEM GENERALISATION
               "MWE_TYPE",  # 11
               "PHENOMENON",  # 10
               "DETERMINER",  # 11
               "EMPM",  # 12
               "EMPP",  # 13
               "COMP",  # 14
               "IWO",  # 15
               "IXP-CREATIVE",  # 16
               "IXP-EXPANSION",  # 17
               "IXP-N",  # 18
               "IXP-W", #19
               "IXP-PUNC", #20
               "MOD",  # 21
               "AGR",  # 22
               "MWO",  # 23
               "VAR",  # 24
               "AN"]  # 25

    def __init__(self):
        s = XlsxToTxt
        start = timeit.default_timer()
         #s.xlsx_to_txt("self", specificFile=11, cellsForTxt=s.cellsForTXT)
        #s.convertAllXlsxToTXT(self)
        s.convertRangeOfXlsxToTXT(self, rangeOfXlsxFilse=[16])
        print(str(round(timeit.default_timer() - start, 3)) + " sec")

    def convertAllXlsxToTXT(self):
        s = XlsxToTxt  # self class
        file_index = 0
        for f in s.Files:
            s.xlsx_to_txt("self", specificFile=file_index, cellsForTxt=s.cellsForTXT)
            file_index += 1

    def convertRangeOfXlsxToTXT(self, rangeOfXlsxFilse=[]):
        s = XlsxToTxt  # self class
        for file_index in rangeOfXlsxFilse:
            s.xlsx_to_txt("self", specificFile=file_index, cellsForTxt=s.cellsForTXT)

    # It convert the scpecific xlsx file to txt
    def xlsx_to_txt(self, specificFile=0, cellsForTxt=[]):
        s = XlsxToTxt  # self class
        if specificFile > 0:
            txtFile = s.txt_datapath + "" + s.Files[specificFile] + ".txt"
            xlsxFile = s.dataset_path + "" + s.Files[specificFile] + ".xlsx"
            with open(txtFile, "w", encoding="utf-8") as wr:
                xlsx = xlrd.open_workbook(xlsxFile, encoding_override='utf-8')
                sheet = xlsx.sheet_by_index(0)  # get a sheet
                flag = 2
                numberOfXlsxRows = sheet.nrows
                txtRow_index = 1
                for rowNum in range(numberOfXlsxRows):
                    xlsxRow_index = rowNum + 1
                    if rowNum == 0:
                        wr.write("FILE # XLSX ROW # TXT ROW ")
                        for h in cellsForTxt:
                            wr.write(" # " + s.headers[h])
                        wr.write("\n")
                    if (xlsxRow_index > 3):
                        flag += 1
                        if flag == 3:
                            txtRow_index += 1
                            wr.write(str(specificFile) + " # " + str(xlsxRow_index) + " # " + str(txtRow_index))
                            for c in cellsForTxt:
                                cellValue = str(sheet.cell(xlsxRow_index - 1, c - 1).value).strip()
                                wr.write(" # " + cellValue)
                                flag = 0
                                if cellValue == "":
                                    print("null cell: " + s.Files[specificFile]
                                          + " Row: " + str(xlsxRow_index)
                                          + " Cell: " + s.headers[c])
                            wr.write("\n")
            wr.close()


XlsxToTxt()
