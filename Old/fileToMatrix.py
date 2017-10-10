import numpy as np

import main as main

class FileToMatrix:
    # it returns the array of vector space
    def fileToVectorSpace(filename):
        vectorSpace = []
        with open(main.dataset_path+""+filename, encoding="utf8") as f:
            count_line = 0
            for line in f:
                count_line = count_line + 1
                if count_line > 1:
                    row = (line.strip()).split("#")
                    for col in row:
                        if col=="":
                            row[col] = "null"
                    vectorSpace.append(row[1:])
        return vectorSpace

        # it returns the array of vector space from the second column and then

    #VectorSpace in whole dataset with usefull attributes only
    def VectorSpace_InWoleDataset_fromSecondColumn_usefulAttr(filenames):
        vectorSpace = []
        feature_names = []
        feature_names.append("Simile_id")
        index_of_file = 0
        flag = 0
        for file in filenames:
            index_of_file += 1
            with open(main.dataset_path + "" + file, encoding="utf8") as f:
                count_line = 0
                for line in f:
                    count_line += 1
                    if count_line == 1 and flag == 0:  # take the names of features
                        flag = 1
                        splited_row = (line.strip()).split("#")
                        col_index = 0
                        for column in splited_row:
                            col_index += 1
                            if col_index > 1 and col_index != 3 and col_index != 6 and col_index != 9:
                                feature_names.append(splited_row[col_index - 1].strip())
                        print(feature_names)
                    if count_line > 1:
                        splited_row = (line.strip()).split("#")
                        col_index = 0
                        rowVector = []
                        rowVector.append(str(index_of_file))
                        for column in splited_row:
                            col_index += 1
                            if col_index > 1 and col_index != 4 and col_index != 6 and col_index != 9:
                                rowVector.append(splited_row[col_index - 1].strip())
                        vectorSpace.append(rowVector)
        return vectorSpace, feature_names


    # it returns the array of vector space from the second column and then as long as the feature names
    # with useful attributes only
    def fileToVectorSpace_from2ndColumn_usefulAttr(filename):
        vectorSpace = []
        feature_names = []
        # print(dataset_path + "" + filename)
        with open(main.dataset_path + "" + filename, encoding="utf8") as f:
            count_line = 0
            for line in f:
                count_line += 1
                if count_line == 1:  # take the names of features
                    splited_row = (line.strip()).split("#")
                    col_index = 0
                    for column in splited_row:
                        col_index += 1
                        if col_index > 1 and col_index != 4 and col_index !=6 and col_index!= 9:
                            feature_names.append(splited_row[col_index - 1].strip())
                    #feature_names.append(splited_row[1:])
                else:
                    splited_row = (line.strip()).split("#")
                    col_index = 0
                    rowVector = []
                    for column in splited_row:
                        col_index += 1
                        if col_index > 1 and col_index != 4 and col_index != 6 and col_index != 9:
                            rowVector.append(splited_row[col_index - 1].strip())
                    vectorSpace.append(rowVector)
        return vectorSpace, feature_names


    # it returns the array of vector space from the second column and then as long as the feature names
    def fileToVectorSpace_fromSecondColumn(filename):
        vectorSpace = []
        feature_names = []
        # print(dataset_path + "" + filename)
        with open(main.dataset_path + "" + filename, encoding="utf8") as f:
            count_line = 0
            for line in f:
                count_line += 1
                if count_line == 1:  # take the names of features
                    splited_row = (line.strip()).split("#")
                    col_index = 0
                    for column in splited_row:
                        col_index += 1
                        if col_index > 1 and col_index != 4 and col_index!= 8:
                            feature_names.append(splited_row[col_index - 1].strip())
                    #feature_names.append(splited_row[1:])
                else:
                    splited_row = (line.strip()).split("#")
                    col_index = 0
                    rowVector = []
                    for column in splited_row:
                        col_index += 1
                        if col_index > 1 and col_index != 4 and col_index != 8:
                            rowVector.append(splited_row[col_index - 1].strip())
                    vectorSpace.append(rowVector)
        return vectorSpace, feature_names


    # it returns the array of vector space from the second column and then
    def VectorSpace_InWoleDataset_fromSecondColumn(filenames):
        vectorSpace = []
        feature_names = []
        feature_names.append("Simile_id")
        index_of_file = 0
        flag = 0
        for file in filenames:
            index_of_file += 1
            with open(main.dataset_path + "" + file, encoding="utf8") as f:
                count_line = 0
                for line in f:
                    count_line += 1
                    if count_line == 1 and flag == 0:  # take the names of features
                        flag = 1
                        splited_row = (line.strip()).split("#")
                        col_index = 0
                        for column in splited_row:
                            col_index += 1
                            if col_index > 1 and col_index != 4 and col_index != 8:
                                feature_names.append(splited_row[col_index - 1].strip())
                    if count_line > 1:
                        splited_row = (line.strip()).split("#")
                        col_index = 0
                        rowVector = []
                        rowVector.append(str(index_of_file))
                        for column in splited_row:
                            col_index += 1
                            if col_index > 1 and col_index != 4 and col_index != 8:
                                rowVector.append(splited_row[col_index - 1].strip())
                        vectorSpace.append(rowVector)
        return vectorSpace, feature_names



    #matrix transposition
    def rowToColTransposition(m):
        rows = len(m)  # rows
        cols = len(m[0])  # columns
        mT = []
        for c in range(0, cols):
            mT.append([])
            for r in range(0, rows):
                mT[c].append(m[r][c])
        return mT


#FileToMatrix.AllfilesToVectorSpace_fromSecondColumn(filenames)
