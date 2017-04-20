import xlrd
import timeit
class XlsxToTxt:





    Files = ["",
                  "1_aspros_san_to_pani",
                  "2_stolismenos_san_fregata",
                  "3_apalos_san_poupoulo",
                  "4_apalos_san_xadi",
                  "5_elafrys_san_poupoulo",
                  "6_kokkinos_san_astakos",
                  "7_oplismenos_san_astakos",
                  "8_malakos_san_voutiro",
                  "9_geros_san_tavros",
                  "10_pistos_san_skilos"]

    linux_dataset_path = "/home/pkouris/Dropbox/EMP_DID_dropbox/NLP/SIMILES/dataset/"
    win_dataset_path = "E:/Dropbox/EMP_DID_dropbox/NLP/SIMILES/dataset/"
    dataset_path = win_dataset_path  #It should be changed according to pc
    txt_datapath = dataset_path + "txt/"

    cellsForTXT = [3, 5, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
    headers = ["",#0
               "TEXT",#1
               "SIMILE",#2
               "GENDER",#3
               "HEAD",#4
               "LEMMA",#5
               "MOD_PRED_SEMS",#6   MODIFIED PRED SEMS
               "TEN_GEN_SEMS",#7  TENOR SEM GENERALISATION
               "MWE_TYPE",#8
               "PHENOMENON", #9
               "DETERMINER",#10
               "EMPM",#11
               "EMPP",#12
               "IWO",#13
               "IXP-CREATIVE",#14
               "IXP-EXPANSION",#15
               "IXP-N_W_PUNC",#16
               "MOD",#17
               "AGR",#18
               "MWO",#19
               "VAR"]#20



    def __init__(self):
        s = XlsxToTxt
        start = timeit.default_timer()
        #s.xlsx_to_txt("self", specificFile=1, cellsForTxt=s.cellsForTXT)
        s.convertAllXlsxToTXT(self)
        #s.convertRangeOfXlsxToTXT(self, rangeOfXlsxFilse=[1,3,5])
        print(str(round(timeit.default_timer()-start, 3)) + " sec")


    def convertAllXlsxToTXT(self):
        s = XlsxToTxt #self class
        file_index = 0
        for f in s.Files:
            s.xlsx_to_txt("self", specificFile=file_index, cellsForTxt=s.cellsForTXT)
            file_index += 1


    def convertRangeOfXlsxToTXT(self, rangeOfXlsxFilse = []):
        s = XlsxToTxt  # self class
        for file_index in rangeOfXlsxFilse:
            s.xlsx_to_txt("self", specificFile=file_index, cellsForTxt=s.cellsForTXT)


    #It convert the scpecific xlsx file to txt
    def xlsx_to_txt(self, specificFile=0, cellsForTxt=[]):
        s = XlsxToTxt  #self class
        if specificFile>0:
            txtFile = s.txt_datapath + "" + s.Files[specificFile] + ".txt"
            xlsxFile = s.dataset_path + "" + s.Files[specificFile] + ".xlsx"
            with open(txtFile, "w", encoding="utf-8") as wr:
                xlsx = xlrd.open_workbook(xlsxFile, encoding_override='utf-8')
                sheet = xlsx.sheet_by_index(0) # get a sheet
                flag = 2
                numberOfXlsxRows = sheet.nrows
                txtRow_index = 1
                for rowNum in range(numberOfXlsxRows):
                    xlsxRow_index = rowNum + 1
                    if rowNum == 0:
                        wr.write("FILE # XLSX ROW # TXT ROW ")
                        for h in cellsForTxt:
                            wr.write("# " + s.headers[h])
                        wr.write("\n")
                    if(xlsxRow_index > 4):
                        flag +=1
                        if flag == 3:
                            txtRow_index += 1
                            wr.write(str(specificFile) + " # " + str(xlsxRow_index) + " # " + str(txtRow_index))
                            for c in cellsForTxt:
                                cellValue = str(sheet.cell(xlsxRow_index-1, c-1).value).strip()
                                wr.write(" # " + cellValue)
                                flag = 0
                                if cellValue=="":
                                    print("null cell:" + s.Files[specificFile]
                                          + "Row: " + str(xlsxRow_index)
                                          + " Cell: " + s.headers[c])
                            wr.write("\n")
            wr.close()




XlsxToTxt()
