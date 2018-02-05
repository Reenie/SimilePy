import numpy as np
import main as main
import VectorSpace_simile as VSsimile

class StatisticsPerFeatureCombination:

    #txt_datapath = main.dataset_path + "txt/"

    def __init__(self):
        s = StatisticsPerFeatureCombination
        #vectorOfDistinctVectors = [['4', '7', '10', '19', '22', '25', '46'],['13', '16'], ['28', '31', '34', '37'],['40'], ['43'], ['49']   ]
        #a = s.entropyOfDistinctVectors(self, vectorOfDistinctVectors, 16)
        #print(a)
        #print(s.log(self))
        #s.distinctNumericalVectorSpace(self)
        s.printStatistics_mweType(self)

    def printStatistics_gender(self):
        s = StatisticsPerFeatureCombination
        results = s.genderDistributionPerPersonInstances(self)
        print("\r\n\r\n")
        print("Gender distribution of PERSON instances")
        print('%-35s%-22s%-19s' % ("File Name", "Instances of Person", "Gender distribution"))
        for r in results:
            sss = "(" + str(r[2][0][0]) + ": " + str(r[2][0][2]) + ")"
            print('\r%-35s%-22s%-19s' % (r[0], sss, r[1]))
        results_inWholeDatasets = s.countValuesOfFeatureValueCombination_inWholeDataset(self)
        for r in results_inWholeDatasets:
            sss = "(" + str(r[2][0][0]) + ": " + str(r[2][0][2]) + ")"
            print('\r%-35s%-22s%-19s' % (r[0], sss, r[1]))




    def printStatistics_mweType(self):
        s = StatisticsPerFeatureCombination
        results = s.semanticsDistributionPerMweType(self)
        print("\r\n\r\n")
        print("Semantic distribution per MWE Type\n")
        for r in results:
            scomp = str(r[1][0]) + "(" + r[1][2] + "): " + str(r[1][3])
            ajsim = str(r[2][0]) + "(" + r[2][2] + "): " + str(r[2][3])
            ss =  str(r[3][0]) + "(" + r[3][2] + "): " + str(r[3][3])
            scon = str(r[4][0]) + "(" + r[4][2] + "): " + str(r[4][3])
            dir = str(r[5][0]) + "(" + r[5][2] + "): " + str(r[5][3])
            indir = str(r[6][0]) + "(" + r[6][2] + "): " + str(r[6][3])
            print(r[0])
            print(scomp)
            print(ajsim)
            print(ss)
            print(scon)
            print(dir)
            print(indir)
            print("\n")
        results_inWhole = s.semanticsDistributionPerMweType_inWholeDataset(self)
        for r in results_inWhole:
                scomp = str(r[1][0]) + "(" + r[1][2] + "): " + str(r[1][3])
                ajsim = str(r[2][0]) + "(" + r[2][2] + "): " + str(r[2][3])
                ss = str(r[3][0]) + "(" + r[3][2] + "): " + str(r[3][3])
                scon = str(r[4][0]) + "(" + r[4][2] + "): " + str(r[4][3])
                dir = str(r[5][0]) + "(" + r[5][2] + "): " + str(r[5][3])
                indir = str(r[6][0]) + "(" + r[6][2] + "): " + str(r[6][3])
                print(r[0])
                print(scomp)
                print(ajsim)
                print(ss)
                print(scon)
                print(dir)
                print(indir)
                print("\n")





    def semanticsDistributionPerMweType(self):
        s = StatisticsPerFeatureCombination
        semanticFeature = "SEMANTICS"
        mweFeature = "MWE_TYPE"
        mweFeatureValeus = ["SCOMP", "AJSIM", "SS", "SCON", "DIR", "INDIR"]
        results = [] #[(filename, [M:freq, F:freq, N:freq])]
        for filename in main.filenames:
            scompDistr = {}
            ajsimDistr = {}
            ssDistr = {}
            sconDistr = {}
            dirDistr = {}
            indirDistr = {}
            categoricalVS, categoricalFeature_names = VSsimile.VectorSpace_simile.featureMatrix("self", [filename], attrForVectorSpace=VSsimile.VectorSpace_simile.attrForVectorSpace)
            ################
            #print(categoricalFeature_names)
            mweFeature_index = -1
            semanticsFeature_index = -1
            index = -1
            for featureName in categoricalFeature_names:
                index += 1
                #print(featureName)
                if featureName == mweFeature:
                    mweFeature_index = index
                if featureName == semanticFeature:
                    semanticsFeature_index = index
            row_count = 0
            for row in categoricalVS:
                row_count += 1
                if row[mweFeature_index]=="SCOMP":
                    if s.hasKey(self, scompDistr, row[semanticsFeature_index]) == 1:
                        newValue = scompDistr[row[semanticsFeature_index]] + 1
                        scompDistr[row[semanticsFeature_index]] += 1
                    else:
                        scompDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index]=="AJSIM":
                    if s.hasKey(self, ajsimDistr, row[semanticsFeature_index]) == 1:
                        newValue = ajsimDistr[row[semanticsFeature_index]] + 1
                        ajsimDistr[row[semanticsFeature_index]] += 1
                    else:
                        ajsimDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index]=="SS":
                    if s.hasKey(self, ssDistr, row[semanticsFeature_index]) == 1:
                        newValue = ssDistr[row[semanticsFeature_index]] + 1
                        ssDistr[row[semanticsFeature_index]] += 1
                    else:
                        ssDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index]=="SCON":
                    if s.hasKey(self, sconDistr, row[semanticsFeature_index]) == 1:
                        newValue = sconDistr[row[semanticsFeature_index]] + 1
                        sconDistr[row[semanticsFeature_index]] += 1
                    else:
                        sconDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index] == "DIR":
                    if s.hasKey(self, dirDistr, row[semanticsFeature_index]) == 1:
                        newValue = dirDistr[row[semanticsFeature_index]] + 1
                        dirDistr[row[semanticsFeature_index]] += 1
                    else:
                        dirDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index] == "INDIR":
                    if s.hasKey(self, indirDistr, row[semanticsFeature_index]) == 1:
                        newValue = indirDistr[row[semanticsFeature_index]] + 1
                        indirDistr[row[semanticsFeature_index]] += 1
                    else:
                        indirDistr[row[semanticsFeature_index]] = 1
            scompDistr_list = []
            ajsimDistr_list = []
            ssDistr_list = []
            sconDistr_list = []
            dirDistr_list = []
            indirDistr_list = []
            for key in scompDistr.keys():
                scompDistr_list.append((key, scompDistr[key], str(round(100*scompDistr[key]/row_count,1))+"%"))
            for key in ajsimDistr.keys():
                ajsimDistr_list.append((key, ajsimDistr[key], str(round(100 * ajsimDistr[key] / row_count, 1)) + "%"))
            for key in ssDistr.keys():
                ssDistr_list.append((key, ssDistr[key], str(round(100 * ssDistr[key] / row_count, 1)) + "%"))
            for key in sconDistr.keys():
                sconDistr_list.append((key, sconDistr[key], str(round(100 * sconDistr[key] / row_count, 1)) + "%"))
            for key in dirDistr.keys():
                dirDistr_list.append((key, dirDistr[key], str(round(100 * dirDistr[key] / row_count, 1)) + "%"))
            for key in indirDistr.keys():
                indirDistr_list.append((key, indirDistr[key], str(round(100 * indirDistr[key] / row_count, 1)) + "%"))
            scompDistr_list.sort(key=lambda tup: -tup[1])
            ajsimDistr_list.sort(key=lambda tup: -tup[1])
            ssDistr_list.sort(key=lambda tup: -tup[1])
            sconDistr_list.sort(key=lambda tup: -tup[1])
            dirDistr_list.sort(key=lambda tup: -tup[1])
            indirDistr_list.sort(key=lambda tup: -tup[1])
            scomp_sum = s.SumOfHashtableValues(self, scompDistr)
            ajsim_sum = s.SumOfHashtableValues(self, ajsimDistr)
            ss_sum = s.SumOfHashtableValues(self, ssDistr)
            scon_sum = s.SumOfHashtableValues(self, sconDistr)
            dir_sum = s.SumOfHashtableValues(self, dirDistr)
            indir_sum = s.SumOfHashtableValues(self, indirDistr)
            scomp_pc = str(round(100*scomp_sum/row_count,1))+"%"
            ajsim_pc = str(round(100*ajsim_sum/row_count,1))+"%"
            ss_pc = str(round(100*ss_sum/row_count,1))+"%"
            scon_pc = str(round(100*scon_sum/row_count,1))+"%"
            dir_pc = str(round(100*dir_sum/row_count,1))+"%"
            indir_pc = str(round(100*indir_sum/row_count,1))+"%"
            results.append((filename, ("SCOMP", scomp_sum, scomp_pc, scompDistr_list), ("AJSIM", ajsim_sum, ajsim_pc, ajsimDistr_list),
                            ("SS", ss_sum, ss_pc, ssDistr_list), ("SCON", scon_sum, scon_pc, sconDistr_list),
                            ("DIR", dir_sum, dir_pc, dirDistr_list), ("INDIR", indir_sum, indir_pc, indirDistr_list)))
        return results




    def semanticsDistributionPerMweType_inWholeDataset(self):
            s = StatisticsPerFeatureCombination
            semanticFeature = "SEMANTICS"
            mweFeature = "MWE_TYPE"
            mweFeatureValeus = ["SCOMP", "AJSIM", "SS", "SCON", "DIR", "INDIR"]
            results = [] #[(filename, [M:freq, F:freq, N:freq])]
            scompDistr = {}
            ajsimDistr = {}
            ssDistr = {}
            sconDistr = {}
            dirDistr = {}
            indirDistr = {}
            categoricalVS, categoricalFeature_names = VSsimile.VectorSpace_simile.featureMatrix("self", main.filenames, attrForVectorSpace=VSsimile.VectorSpace_simile.attrForVectorSpace)
            ################
            #print(categoricalFeature_names)
            mweFeature_index = -1
            semanticsFeature_index = -1
            index = -1
            for featureName in categoricalFeature_names:
                index += 1
                #print(featureName)
                if featureName == mweFeature:
                    mweFeature_index = index
                if featureName == semanticFeature:
                    semanticsFeature_index = index
            row_count = 0
            for row in categoricalVS:
                row_count += 1
                if row[mweFeature_index]=="SCOMP":
                    if s.hasKey(self, scompDistr, row[semanticsFeature_index]) == 1:
                        newValue = scompDistr[row[semanticsFeature_index]] + 1
                        scompDistr[row[semanticsFeature_index]] += 1
                    else:
                        scompDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index]=="AJSIM":
                    if s.hasKey(self, ajsimDistr, row[semanticsFeature_index]) == 1:
                        newValue = ajsimDistr[row[semanticsFeature_index]] + 1
                        ajsimDistr[row[semanticsFeature_index]] += 1
                    else:
                        ajsimDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index]=="SS":
                    if s.hasKey(self, ssDistr, row[semanticsFeature_index]) == 1:
                        newValue = ssDistr[row[semanticsFeature_index]] + 1
                        ssDistr[row[semanticsFeature_index]] += 1
                    else:
                        ssDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index]=="SCON":
                    if s.hasKey(self, sconDistr, row[semanticsFeature_index]) == 1:
                        newValue = sconDistr[row[semanticsFeature_index]] + 1
                        sconDistr[row[semanticsFeature_index]] += 1
                    else:
                        sconDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index] == "DIR":
                    if s.hasKey(self, dirDistr, row[semanticsFeature_index]) == 1:
                        newValue = dirDistr[row[semanticsFeature_index]] + 1
                        dirDistr[row[semanticsFeature_index]] += 1
                    else:
                        dirDistr[row[semanticsFeature_index]] = 1
                if row[mweFeature_index] == "INDIR":
                    if s.hasKey(self, indirDistr, row[semanticsFeature_index]) == 1:
                        newValue = indirDistr[row[semanticsFeature_index]] + 1
                        indirDistr[row[semanticsFeature_index]] += 1
                    else:
                        indirDistr[row[semanticsFeature_index]] = 1
            scompDistr_list = []
            ajsimDistr_list = []
            ssDistr_list = []
            sconDistr_list = []
            dirDistr_list = []
            indirDistr_list = []
            for key in scompDistr.keys():
                scompDistr_list.append((key, scompDistr[key], str(round(100*scompDistr[key]/row_count,1))+"%"))
            for key in ajsimDistr.keys():
                ajsimDistr_list.append((key, ajsimDistr[key], str(round(100 * ajsimDistr[key] / row_count, 1)) + "%"))
            for key in ssDistr.keys():
                ssDistr_list.append((key, ssDistr[key], str(round(100 * ssDistr[key] / row_count, 1)) + "%"))
            for key in sconDistr.keys():
                sconDistr_list.append((key, sconDistr[key], str(round(100 * sconDistr[key] / row_count, 1)) + "%"))
            for key in dirDistr.keys():
                dirDistr_list.append((key, dirDistr[key], str(round(100 * dirDistr[key] / row_count, 1)) + "%"))
            for key in indirDistr.keys():
                indirDistr_list.append((key, indirDistr[key], str(round(100 * indirDistr[key] / row_count, 1)) + "%"))
            scompDistr_list.sort(key=lambda tup: -tup[1])
            ajsimDistr_list.sort(key=lambda tup: -tup[1])
            ssDistr_list.sort(key=lambda tup: -tup[1])
            sconDistr_list.sort(key=lambda tup: -tup[1])
            dirDistr_list.sort(key=lambda tup: -tup[1])
            indirDistr_list.sort(key=lambda tup: -tup[1])
            scomp_sum = s.SumOfHashtableValues(self, scompDistr)
            ajsim_sum = s.SumOfHashtableValues(self, ajsimDistr)
            ss_sum = s.SumOfHashtableValues(self, ssDistr)
            scon_sum = s.SumOfHashtableValues(self, sconDistr)
            dir_sum = s.SumOfHashtableValues(self, dirDistr)
            indir_sum = s.SumOfHashtableValues(self, indirDistr)
            scomp_pc = str(round(100*scomp_sum/row_count,1))+"%"
            ajsim_pc = str(round(100*ajsim_sum/row_count,1))+"%"
            ss_pc = str(round(100*ss_sum/row_count,1))+"%"
            scon_pc = str(round(100*scon_sum/row_count,1))+"%"
            dir_pc = str(round(100*dir_sum/row_count,1))+"%"
            indir_pc = str(round(100*indir_sum/row_count,1))+"%"
            results.append(("In Whole Dataset", ("SCOMP", scomp_sum, scomp_pc, scompDistr_list), ("AJSIM", ajsim_sum, ajsim_pc, ajsimDistr_list),
                            ("SS", ss_sum, ss_pc, ssDistr_list), ("SCON", scon_sum, scon_pc, sconDistr_list),
                            ("DIR", dir_sum, dir_pc, dirDistr_list), ("INDIR", indir_sum, indir_pc, indirDistr_list)))
            return results






    def genderDistributionPerPersonInstances(self):
        ###################
        featureNamesCombination = ["SEMANTICS"]
        featureValuesCombination = ["PERSON"]
        TargetFeature = "GENDER"
        targetFeatureValeus = ["M", "F", "N"]
        if VSsimile.VectorSpace_simile.numOfGenders == 2:
            TargetFeatureValeus = ["M/F", "N"]
        ##############
        results = [] #[(filename, [M:freq, F:freq, N:freq])]
        for filename in main.filenames:
            row_count = 0
            featureValues_count = {}
            targetFeatureValues_frequency = {}
            for i in targetFeatureValeus:
                targetFeatureValues_frequency[i] = 0
            for i in featureValuesCombination:
                featureValues_count[i] = 0
            categoricalVS, categoricalFeature_names = VSsimile.VectorSpace_simile.featureMatrix("self", [filename], attrForVectorSpace=VSsimile.VectorSpace_simile.attrForVectorSpace)
            targetFeature_index = -1
            featereNamesCombination_index = []
            index = -1
            for featureName in categoricalFeature_names:
                index += 1
                #print(featureName)
                if featureName == TargetFeature:
                    targetFeature_index = index
                if featureName == featureNamesCombination[0]:
                    featereNamesCombination_index.append(index)
            person_count = 0
            for row in categoricalVS:
                row_count += 1
                if row[featereNamesCombination_index[0]] == featureValuesCombination[0]:
                    person_count += 1
                    gender = row[targetFeature_index]
                    odlValue = targetFeatureValues_frequency[gender]
                    newValue = odlValue + 1
                    targetFeatureValues_frequency[gender] = newValue
            temp_list = []
            featuerValuesCount_list = []
            for i in targetFeatureValeus:
                temp_list.append((i, targetFeatureValues_frequency[i], str(round(100*targetFeatureValues_frequency[i]/person_count,1))+"%"))
            for i in featureValuesCombination:
                featuerValuesCount_list.append((i, person_count, str(round(100*person_count/row_count, 1))+"%"))
            results.append((filename, temp_list, featuerValuesCount_list, row_count))
        return results




    def genderDistributionPerPersonInstances_inWholeDataset(self):
        featureNamesCombination = ["SEMANTICS"]
        featureValuesCombination = ["PERSON"]
        TargetFeature = "GENDER"
        targetFeatureValeus = ["M", "F", "N"]
        results = [] #[(filename, [M:freq, F:freq, N:freq])]
        if VSsimile.VectorSpace_simile.numOfGenders == 2:
            TargetFeatureValeus = ["M/F", "N"]
        row_count = 0
        featureValues_count = {}
        targetFeatureValues_frequency = {}
        for i in targetFeatureValeus:
            targetFeatureValues_frequency[i] = 0
        for i in featureValuesCombination:
            featureValues_count[i] = 0
        categoricalVS, categoricalFeature_names = VSsimile.VectorSpace_simile.featureMatrix("self", main.filenames, attrForVectorSpace=VSsimile.VectorSpace_simile.attrForVectorSpace)
        targetFeature_index = -1
        featereNamesCombination_index = []
        index = -1
        for featureName in categoricalFeature_names:
            index += 1
            #print(featureName)
            if featureName == TargetFeature:
                    targetFeature_index = index
            if featureName == featureNamesCombination[0]:
                    featereNamesCombination_index.append(index)
        person_count = 0
        for row in categoricalVS:
           row_count += 1
           if row[featereNamesCombination_index[0]] == featureValuesCombination[0]:
                    person_count += 1
                    gender = row[targetFeature_index]
                    odlValue = targetFeatureValues_frequency[gender]
                    newValue = odlValue + 1
                    targetFeatureValues_frequency[gender] = newValue

            #temp_dict = targetFeatureValues_frequency.items()
        temp_list = []
        featuerValuesCount_list = []
        for i in targetFeatureValeus:
                temp_list.append((i, targetFeatureValues_frequency[i], str(round(100*targetFeatureValues_frequency[i]/person_count,1))+"%"))
        for i in featureValuesCombination:
                featuerValuesCount_list.append((i, person_count, str(round(100*person_count/row_count, 1))+"%"))
        results.append(("In Whole Dataset", temp_list, featuerValuesCount_list, row_count))
        return results


    def hasKey(self, hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0

    def SumOfHashtableValues(self, hashTable):
        sum = 0
        for k in hashTable.keys():
            sum += hashTable[k]
        return sum


StatisticsPerFeatureCombination()