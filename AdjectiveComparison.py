import xlrd
from sklearn.metrics.pairwise import cosine_similarity
import VectorSpace_simile as VSS
import main as main
import numpy as np
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt

class AdjectiveComparison:
    apalos =("apalos_comparative.xlsx",
                       ["3_apalos_san_poupoulo.2.txt",
                        "4_apalos_san_xadi.2.txt"], 241)

    aspros =  ("aspros_comparative.xlsx",
                         ["1_aspros_san_to_pani.2.txt",
                         "15_aspros_san_to_gala.2.txt",
                         "20_aspros_san_to_xioni.2.txt"], 200)

    grigoros = ("grigoros_comparative.xlsx",
                          ["17_grigoros_san_astrapi.2.txt"], 216)

    kokkinos = ("kokkinos_comparative.xlsx",
                          ["6_kokkinos_san_astakos.2.txt",
                           "11_kokkinos_san_paparouna.2.txt",
                           "13_kokkinos_san_to_pantzari.2.txt"
                          ], 200)

    pistos = ("pistos_comparative.xlsx",
                        ["10_pistos_san_skilos.2.txt"], 200)


    elafrys = ("elafrys_comparative.xlsx", ["5_elafrys_san_poupoulo.2.txt"], 203)
    geros = ("geros_comparative.xlsx", ["9_geros_san_tavros.2.txt"], 200)
    glikos = ("glikos_comparative.xlsx", ["14_glikos_san_meli.2.txt"], 204)
    kryos = ("kryos_comparative.xlsx", ["16_krios_san_ton_pago.2.txt"], 200)
    mavros = ("mavros_comparative.xlsx", ["18_mavros_san_skotadi.2.txt"], 203)

    malakos = ("malakos_comparative.xlsx", ["8_malakos_san_voutiro.2.txt"], 203)
    stolismenos = ("stolismenos_comparative.xlsx", ["2_stolismenos_san_fregata.2.txt"], 182)
    ntimenos = ("ntimenos_comparative.xlsx", ["12_ntimenos_san_astakos.2.txt"], 200)
    mperdemenos = ("mperdemenos_comparative.xlsx", ["19_mperdemenos_san_to_kouvari.2.txt"], 200)
    oplismenos = ("oplismenos_comparative.xlsx", ["7_oplismenos_san_astakos.2.txt"], 240)



    filenames = [apalos, aspros, elafrys, geros, glikos, grigoros, kokkinos, kryos, malakos,
                 mavros, mperdemenos, ntimenos, oplismenos, pistos, stolismenos]



    filenames_ = ["0",
                 "1_aspros_san_to_pani.2.txt",
                 "2_stolismenos_san_fregata.2.txt",
                 "3_apalos_san_poupoulo.2.txt",
                 "4_apalos_san_xadi.2.txt",
                 "5_elafrys_san_poupoulo.2.txt",
                 "6_kokkinos_san_astakos.2.txt",
                 "7_oplismenos_san_astakos.2.txt",
                 "8_malakos_san_voutiro.2.txt",
                 "9_geros_san_tavros.2.txt",
                 "10_pistos_san_skilos.2.txt",
                 "11_kokkinos_san_paparouna.2.txt",
                 "12_ntimenos_san_astakos.2.txt",
                 "13_kokkinos_san_to_pantzari.2.txt",
                 "14_glikos_san_meli.2.txt",
                 "15_aspros_san_to_gala.2.txt",
                 "16_krios_san_ton_pago.2.txt",
                 "17_grigoros_san_astrapi.2.txt",
                 "18_mavros_san_skotadi.2.txt",
                 "19_mperdemenos_san_to_kouvari.2.txt",
                 "20_aspros_san_to_xioni.2.txt"
                 ]

    def __init__(self):
        s = AdjectiveComparison
        s.printReport(self, filenames=s.filenames)
        s.charts(self)
        s.charts_allSimilesComparison(self)


        #print(cosine_similarity([[0.7, 0.3, 0]], [[0.1, 0.8, 0.1]]))
        #print(cosine_similarity([[1, 1, 0]], [[1, 1, 1]]))
        #s.cosineSimilarity(self, simileFilenames=s.filenames_aspros[1], xlsx_filename=s.filenames_aspros[0], maxXlsxRows=200)



    def printReport(self, filenames=filenames):
        s = AdjectiveComparison
        s.cosineSimilarityCalAndPrint(self, filenames=filenames)
        s.entropyCalAndPrint(self, filenames=filenames)
        s.valuesOfSimileAndFreeAdjPrint(self, filenames=filenames)
        #s.charts(self)
        #s.vectorsForMeasuringSimilarityPrint(self, filenames=filenames)

    def charts(self):
        s = AdjectiveComparison
        s.charts(self)
        s.charts_allSimilesComparison(self)
        #plt.show()


    def charts(self):
        s = AdjectiveComparison
        similarity = []
        #print("\n\n\nValues")
        for fn in s.filenames:
            simileAdj = []
            simileAdjLineName = []
            allSimileAdj = []
            allSimileAdjLineName = []
            freeAdj = []
            freeAdjLineName = []
            wordNetNames = []
            for f in fn[1]:
                a = s.cosineSimilarity(self, simileFilenames=[f], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                simileAdj.append(a[1][2])
                simileAdjLineName.append(f.replace(".2.txt", ""))
                newNames = s.extractNamesOfSems(self, a[1][2])
                wordNetNames = s.combineWordNetNames(self, wordNetNames, newNames)
            #if len(fn[1]) > 1:
                #aa = s.cosineSimilarity(self, simileFilenames=fn[1], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                #allSimileAdj.append(aa[1][2])
                #allSimileAdjLineName.append("All similes of " + fn[0].replace("_comparative.xlsx", ""))
            freeAdj.append(a[1][3])
            freeAdjLineName.append(fn[0].replace("_comparative.xlsx", "") + " (free adj)")
            newNames = s.extractNamesOfSems(self, a[1][3])
            wordNetNames = s.combineWordNetNames(self, wordNetNames, newNames)

            x_values, y_values, lineNames = s.adjDistributionCharts(self, simileAdj, allSimileAdj, freeAdj, simileAdjLineName, allSimileAdjLineName, freeAdjLineName, wordNetNames)
            #print(fn[0].replace("_comparative.xlsx", ""))
            #print(wordNetNames)
            #print(len(wordNetNames))
            plt.style.use('ggplot')
            fig = plt.figure(figsize=(9, 5), dpi=150, edgecolor='b', frameon='false')
            size = len(x_values)
            ind = np.arange(size)
            width = 0.3  #(1. - 2. * margin) / size
            margin = width * len(lineNames) * 2
            if size>3:
                width = 0.20
                margin = width * len(lineNames) * 3
            elif size == 2:
                width = 0.25
                margin = width * len(lineNames) * 2.5
            ax = plt.subplot(1, 1, 1)
            num = -1
            color = ['#1a75ff', '#ffaf1a', '#00ff00', '#ff0066', 'm', 'c', 'k', 'y']
            margin = width*len(lineNames)*2.5
            for y_v, label in zip(y_values, lineNames):
                num += 1
                #print("plotting: ", y_v)
                x_pos = ind + (num * width) + margin
                gene_rects = plt.bar(x_pos, y_v, width, label=label, color=color[num], align="center")
            ax.set_ylabel('Frequency (%)')
            ax.set_xlabel('Entity')
            plt.title(fn[0].replace("_comparative.xlsx", ""))
            ax.set_xticks(ind + width*num/2 + margin)
            ax.set_xticklabels(x_values, rotation='vertical')
            ax.legend()
            plt.tight_layout()
            fig.savefig("Figures/" + fn[0].replace("_comparative.xlsx", "") + ".pdf")
            fig.savefig("Figures/" + fn[0].replace("_comparative.xlsx", "") + ".png")
        #plt.show()

    #comparison between all similes and free adj
    def charts_allSimilesComparison(self):
        s = AdjectiveComparison
        #print("\n\n\nValues")
        for fn in s.filenames:
            simileAdj = []
            simileAdjLineName = []
            allSimileAdj = []
            allSimileAdjLineName = []
            freeAdj = []
            freeAdjLineName = []
            wordNetNames = []
            if len(fn[1]) > 1:
                aa = s.cosineSimilarity(self, simileFilenames=fn[1], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                allSimileAdj.append(aa[1][2])
                allSimileAdjLineName.append("All similes of " + fn[0].replace("_comparative.xlsx", "") )
                newNames = s.extractNamesOfSems(self, aa[1][2])
                wordNetNames = s.combineWordNetNames(self, wordNetNames, newNames)
                freeAdj.append(aa[1][3])
                freeAdjLineName.append(fn[0].replace("_comparative.xlsx", "") + " (free adj)")
                newNames = s.extractNamesOfSems(self, aa[1][3])
                wordNetNames = s.combineWordNetNames(self, wordNetNames, newNames)

                x_values, y_values, lineNames = s.adjDistributionCharts(self, simileAdj, allSimileAdj, freeAdj,
                                                                    simileAdjLineName, allSimileAdjLineName,
                                                                    freeAdjLineName, wordNetNames)
                #print(fn[0].replace("_comparative.xlsx", ""))
                #print(wordNetNames)
                #print(len(wordNetNames))
                plt.style.use('ggplot')
                fig = plt.figure(figsize=(9, 5), dpi=150, edgecolor='b', frameon='True')
                size = len(x_values)
                ind = np.arange(size)
                width = 0.3  # (1. - 2. * margin) / size
                margin = width * len(lineNames) * 2
                if size > 3:
                    width = 0.20
                    margin = width * len(lineNames) * 3
                elif size == 2:
                    width = 0.25
                    margin = width * len(lineNames) * 2.5
                ax = plt.subplot(1, 1, 1)
                num = -1
                color = ['#1a75ff', '#ffaf1a', '#00ff00', '#ff0066', 'm', 'c', 'k', 'y']
                margin = width * len(lineNames) * 2.5
                for y_v, label in zip(y_values, lineNames):
                    num += 1
                    #print("plotting: ", y_v)
                    x_pos = ind + (num * width) + margin
                    gene_rects = plt.bar(x_pos, y_v, width, label=label, color=color[num], align="center")
                ax.set_ylabel('Frequency (%)')
                ax.set_xlabel('Entity')
                plt.title(fn[0].replace("_comparative.xlsx", "") + " (all similes)")
                ax.set_xticks(ind + width * num / 2 + margin)
                ax.set_xticklabels(x_values, rotation='vertical')
                ax.legend()
                plt.tight_layout()
                fig.savefig("Figures/" + fn[0].replace("_comparative.xlsx", "") + " (all similes).png",  bbox_inches='tight')
                fig.savefig("Figures/" + fn[0].replace("_comparative.xlsx", "") + " (all similes).pdf",  bbox_inches='tight')
        #plt.show()

    def combineWordNetNames(self, oldNames, newNames):
        for n in newNames:
            if n not in oldNames:
                oldNames.append(n)
        oldNames.sort()
        return oldNames

    #It returns the wordNet names (i.e PERSON, ANIMAL, LOCATION)
    def extractNamesOfSems(self, semanticDistribution):
        s = AdjectiveComparison
        names = []
        for i in semanticDistribution:
            names.append(i[0])
        return names


    def adjDistributionCharts(self, simileAdj, allSimeleAdj, freeAdj, simileAdjLineName, allSimileAdjLineName, freeAdjLineName, wordNetNames):
        s = AdjectiveComparison
        x_values = wordNetNames
        #print(x_values)
        size = len(x_values)
        y_values = []
        lines = []
        for sa, nn in zip(freeAdj, freeAdjLineName):
            #print(sa)
            freq_temp = [0.0] * size
            for ss in sa:
                index = -1
                #print(ss)
                #print(ss[0])
                for x in x_values:
                    index += 1
                    if x == ss[0]:
                        freq_temp[index] = ss[1]*100.0
            y_values.append(freq_temp)
            lines.append(nn)
        if len(allSimeleAdj)>0:
            for sa, nn in zip(allSimeleAdj, allSimileAdjLineName):
                # print(sa)
                freq_temp = [0.0] * size
                for ss in sa:
                    index = -1
                    # print(ss)
                    # print(ss[0])
                    for x in x_values:
                        index += 1
                        if x == ss[0]:
                            freq_temp[index] = ss[1]*100.0
                y_values.append(freq_temp)
                lines.append(nn)
        for sa, nn in zip(simileAdj, simileAdjLineName):
            #print(sa)
            freq_temp = [0.0] * size
            for ss in sa:
                index = -1
                #print(ss)
                #print(ss[0])
                for x in x_values:
                    index += 1
                    if x == ss[0]:
                        freq_temp[index] = ss[1]*100.0
            y_values.append(freq_temp)
            lines.append(nn)
        return x_values, y_values, lines





    def cosineSimilarityCalAndPrint(self, filenames = []):
        s = AdjectiveComparison
        binarySimilarity = []
        freqSimilarity = []
        similarityBetween = []
        print("Similarity between simile adjective and free adjective")
        print('%-40s%-38s%-40s' % ("Simile - Free adjective", "Cosine Similarity based on frequency", "Cosine Similarity based on binary vectors"))
        for fn in s.filenames:
            for f in fn[1]:
                a = s.cosineSimilarity(self, simileFilenames = [f], xlsx_filename = fn[0] , maxXlsxRows=fn[2])
                #similarity.append((f, fn[0], a))
                #print(f +" - " + fn[0])
                print('%-40s%-38s%-40s' % (f.replace(".2.txt", "") +" - " + fn[0].replace("_comparative.xlsx", ""), round(a[0][0][0][0], 3), round(a[0][2][0][0], 3)))
                binarySimilarity.append(a[0][2][0][0])
                freqSimilarity.append(a[0][0][0][0])
                similarityBetween.append(f.replace(".2.txt", ""))
            if len(fn[1]) > 1:
                aa = s.cosineSimilarity(self, simileFilenames=fn[1], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                #similarity.append(("All - ", fn[0], aa))
                print('%-40s%-38s%-40s' %
                      ("All " + fn[0].replace("_comparative.xlsx", "") + " similes - "
                       + fn[0].replace("_comparative.xlsx", ""), round(aa[0][0][0][0], 3), round(aa[0][2][0][0], 3)))
            print('%-40s%-38s%-40s' % ("--------------------------------------", "-----", "-----"))
            s.chart_similarity(self, similarityBetween, binarySimilarity, freqSimilarity)


    def entropyCalAndPrint(self, filenames=[]):
        s = AdjectiveComparison
        #similarity = []
        print("\n\n\nEntropy as a measure of diversity:")
        print('%-40s%-30s%-30s' % (
        "Simile - Free adjective", "Entropy of simile adjective", "Entropy of free adjective"))
        simileEntropy= []
        freeAdjEntropy = []
        similarityBetween = []
        for fn in s.filenames:
            for f in fn[1]:
                a = s.cosineSimilarity(self, simileFilenames=[f], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                #similarity.append((f, fn[0], a))
                # print(f +" - " + fn[0])
                print('%-40s%-30s%-30s' % (
                f.replace(".2.txt", "") + " - " + fn[0].replace("_comparative.xlsx", ""),
                round(a[1][0], 3), round(a[1][1], 3)))
                similarityBetween.append(f.replace(".2.txt", ""))
                simileEntropy.append(a[1][0])
                freeAdjEntropy.append(a[1][1])
            if len(fn[1]) > 1:
                aa = s.cosineSimilarity(self, simileFilenames=fn[1], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                #similarity.append(("All - ", fn[0], aa))
                print('%-40s%-30s%-30s' %
                      ("All " + fn[0].replace("_comparative.xlsx", "") + " similes - "
                       + fn[0].replace("_comparative.xlsx", ""),
                       round(aa[1][0], 3), round(aa[1][1], 3)))
            print('%-40s%-30s%-30s' % ("--------------------------------------", "-----", "-----"))
        pearson_correlation = pearsonr(simileEntropy, freeAdjEntropy)
        # print(pearsonr([1, 2, 3, 4, 5, 6], [-6, -5, -4, -3, -2, -1]))
        print("\nPearson_correlation(Simile adjective entropy, Free adjective entropy) = " + str(
                round(pearson_correlation[0], 4)) +
                  ",  p-value = " + str(round(pearson_correlation[1], 4)))
        s.chart_entropy(self, similarityBetween, simileEntropy, freeAdjEntropy)
        #print(simileEntropy)
        #print(freeAdjEntropy)


    def chart_similarity(self, similarityBetween = [], binarySimilarity = [], freqSimilarity = []):
        s = AdjectiveComparison
        x_values = similarityBetween
        y_values = [freqSimilarity, binarySimilarity]
        lineNames = ["Similarity based on frequency", "Similarity based on binary vectors"]
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(9, 5), dpi=150, edgecolor='b', frameon='false')
        size = len(x_values)
        ind = np.arange(size)
        width = 0.3  # (1. - 2. * margin) / size
        margin = width * len(lineNames) * 2.5
        ax = plt.subplot(1, 1, 1)
        num = -1
        color = ['#00ff00', '#ff0066', '#ffaf1a', '#1a75ff', 'm', 'c', 'k', 'y']
        for y_v, label in zip(y_values, lineNames):
                num += 1
                x_pos = ind + (num * width) + margin
                gene_rects = plt.bar(x_pos, y_v, width, label=label, color=color[num], align="center")
        ax.set_ylabel('Cosine Similarity')
        ax.set_xlabel('Simile')
        plt.title("Similarity between entities of simile and free adjective")
        ax.set_xticks(ind + width * num / 2 + margin)
        ax.set_xticklabels(x_values, rotation='vertical')
        ax.legend()
        plt.tight_layout()
        fig.savefig("Figures/" + "Chart_Similarity" + ".pdf")
        fig.savefig("Figures/" + "Chart_Similarity" + ".png")

    def chart_entropy(self, entropyOf = [], simileEntropy = [], freeAdjEntropy = []):
        s = AdjectiveComparison
        x_values = entropyOf
        y_values = [simileEntropy, freeAdjEntropy]
        lineNames = ["Simile entities", "Free adj entities"]
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(9, 5), dpi=150, edgecolor='b', frameon='false')
        size = len(x_values)
        ind = np.arange(size)
        width = 0.3  # (1. - 2. * margin) / size
        margin = width * len(lineNames) * 2.5
        ax = plt.subplot(1, 1, 1)
        num = -1
        color = ['#ffaf1a', '#1a75ff',  '#00ff00', '#ff0066', 'm', 'c', 'k', 'y']
        for y_v, label in zip(y_values, lineNames):
                num += 1
                x_pos = ind + (num * width) + margin
                gene_rects = plt.bar(x_pos, y_v, width, label=label, color=color[num], align="center")
        ax.set_ylabel('Entropy')
        ax.set_xlabel('Simile')
        plt.title("Diversity of simile and free adjective entities")
        ax.set_xticks(ind + width * num / 2 + margin)
        ax.set_xticklabels(x_values, rotation='vertical')
        ax.legend()
        plt.tight_layout()
        fig.savefig("Figures/" + "Chart_Diversity" + ".pdf")
        fig.savefig("Figures/" + "Chart_Diversity" + ".png")


    def valuesOfSimileAndFreeAdjPrint(self, filenames=[]):
        s = AdjectiveComparison
        similarity = []
        print("\n\n\nValues")
        # print('%-40s%-30s%-30s' % (
        # "Simile - Free adjective", "Entropy(adjective of simile)", "Entropy(free adjective)"))
        for fn in s.filenames:
            for f in fn[1]:
                a = s.cosineSimilarity(self, simileFilenames=[f], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                print(f.replace(".2.txt", ""), end="\n   ")
                print(a[1][2])
            if len(fn[1]) > 1:
                aa = s.cosineSimilarity(self, simileFilenames=fn[1], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                print("All similes of " + fn[0].replace("_comparative.xlsx", ""), end="\n  ")
                print(aa[1][2])
            print("Free " + fn[0].replace("_comparative.xlsx", ""), end="\n   ")
            print(a[1][3])
            print("------------------------------------------------")




    def vectorsForMeasuringSimilarityPrint(self, filenames=[]):
        s = AdjectiveComparison
        similarity = []
        print("\n\n\nVectors for Measuring Similarity between simile adjective and free adjective")
        #print('%-40s%-30s%-30s' % (
        #"Simile - Free adjective", "Entropy(adjective of simile)", "Entropy(free adjective)"))
        for fn in s.filenames:
            for f in fn[1]:
                a = s.cosineSimilarity(self, simileFilenames=[f], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                #similarity.append((f, fn[0], a))
                # print(f +" - " + fn[0])
                temp_str = ""
                l = len(a[3])
                for i in range(l):
                    temp_str = "%-15s"
                names = tuple(a[3])
                print("%-15s" + temp_str % ("   ",) + str(names))
                print("%-15s" + temp_str % ("All " + fn[0].replace("_comparative.xlsx", "") + " similes",) + str(tuple(a[2][1])))
                print("%-15s" + temp_str % (fn[0].replace("_comparative.xlsx", ""),) + str(tuple(a[2][3])))
            if len(fn[1]) > 1:
                aa = s.cosineSimilarity(self, simileFilenames=fn[1], xlsx_filename=fn[0], maxXlsxRows=fn[2])
                #similarity.append(("All - ", fn[0], aa))
                temp_str = ""
                l = len(a[3])
                for i in range(l):
                    temp_str = "%-15s"
                names = tuple(a[3])
                print("%-15s" + temp_str % ("   ",) + str(names))
                print("%-15s" + temp_str % (f.replace(".2.txt", ""),) + str(tuple(a[2][1])))
                print("%-15s" + temp_str % (fn[0].replace("_comparative.xlsx", ""),) + str(tuple(a[2][3])))
            print('%-40s%-30s%-30s' % ("--------------------------------------", "-----", "-----"))



    def cosineSimilarity(self, simileFilenames=[], xlsx_filename="", maxXlsxRows=200):
        s = AdjectiveComparison
        simileSimilarityVector_fraction, simileSimilarityVector_freq, \
        freeAdjSimilarityVector_fraction, freeAdjSimilarityVector_freq,\
        simileSimilarityVector_binary, freeAdjSimilarityVector_binary,\
        entr_simile, entr_freeAdj, freq_simile, freq_freeAdj, unionOfValues\
            = s.similarityVector(self, simileFilenames=simileFilenames, xlsx_filename=xlsx_filename, maxXlsxRows=200)
        cos_sim_frac = cosine_similarity([simileSimilarityVector_fraction], [freeAdjSimilarityVector_fraction])
        cos_sim_freq = cosine_similarity([simileSimilarityVector_freq], [freeAdjSimilarityVector_freq])
        cos_sim_bin = cosine_similarity([simileSimilarityVector_binary], [freeAdjSimilarityVector_binary])
        #cos_sim3 = cosine_similarity([1, 2, 2, 3, 3, 4], [1, 2, 2, 3, 3, 4])
        #print(cos_sim_frac, end="\n\n")
        #print(cos_sim_bin, end="\n\n")
        #print(cos_sim_freq, end="\n\n")
        #print(cos_sim3, end="\n\n")
        return ((cos_sim_frac, cos_sim_freq, cos_sim_bin),
                (entr_simile, entr_freeAdj, freq_simile, freq_freeAdj),
                (simileSimilarityVector_fraction, simileSimilarityVector_freq, \
                freeAdjSimilarityVector_fraction, freeAdjSimilarityVector_freq,
                simileSimilarityVector_binary, freeAdjSimilarityVector_binary), unionOfValues)



    def similarityVector(self, simileFilenames = [], xlsx_filename = "", maxXlsxRows=200):
        s = AdjectiveComparison
        uniqueSimileValues, entr_simile, freq_simile = s.SimileTermFrequencies(self, filenames=simileFilenames)
        uniqueFreeAdjValues, entr_freeAdj, freq_freeAdj = s.freeAdjTermFrequencies(self, xlsx_filename=xlsx_filename, maxXlsxRows=maxXlsxRows)
        unionOfValues = set(uniqueSimileValues) | set(uniqueFreeAdjValues)
        simileSimilarityVector_freq = []
        freeAdjSimilarityVector_freq = []
        simileSimilarityVector_fraction = []
        freeAdjSimilarityVector_fraction = []
        simileSimilarityVector_binary = []
        freeAdjSimilarityVector_binary = []
        for u in unionOfValues:
            v1 = s.valueExistsInFreq(self, u, freq_simile)
            v2 = s.valueExistsInFreq(self, u, freq_freeAdj)
            simileSimilarityVector_fraction.append(v1[0])
            simileSimilarityVector_freq.append(v1[1])
            freeAdjSimilarityVector_fraction.append(v2[0])
            freeAdjSimilarityVector_freq.append(v2[1])
            if v1[1]> 0 :
                simileSimilarityVector_binary.append(1)
            else:
                simileSimilarityVector_binary.append(0)
            if v2[1]> 0 :
                freeAdjSimilarityVector_binary.append(1)
            else:
                freeAdjSimilarityVector_binary.append(0)
        #print(unionOfValues, end="\n\n")
        #print(simileSimilarityVector_freq, end="\n\n")
        #print(freeAdjSimilarityVector_freq, end="\n\n")
        #print(simileSimilarityVector_fraction, end="\n\n")
        #print(freeAdjSimilarityVector_fraction, end="\n\n")
        #print(simileSimilarityVector_binary, end="\n\n")
        #print(freeAdjSimilarityVector_binary, end="\n\n")
        return simileSimilarityVector_fraction, simileSimilarityVector_freq, \
               freeAdjSimilarityVector_fraction, freeAdjSimilarityVector_freq, \
               simileSimilarityVector_binary, freeAdjSimilarityVector_binary, \
               entr_simile, entr_freeAdj, freq_simile, freq_freeAdj, unionOfValues




    def valueExistsInFreq(self, value, freq):
        a = AdjectiveComparison
        for f in freq:
            if value == f[0]:
                return (f[1], f[2])
        return (0, 0)

    def SimileTermFrequencies(self, filenames = []):
        s = AdjectiveComparison
        vss = VSS.VectorSpace_simile
        simileMatrix, _ = vss.featureMatrix(self, filenames=filenames, attrForVectorSpace=[9])
        simileValues = []
        for m in simileMatrix:
            simileValues.append(m[0])
        #print(matrix, end="\n\n")
        #print(vector, end="\n\n")
        entr, freq = s.entropy(self, simileValues)
        uniqueSimileValues = list(set(simileValues))
        #print(uniqueSimileValues, end="\n\n")
        #print(entr, end="\n\n")
        #print(freq, end="\n\n")
        return uniqueSimileValues, entr, freq

    def freeAdjTermFrequencies(self,  xlsx_filename = "", maxXlsxRows = 200):
        s = AdjectiveComparison  # self class
        xlsxFile = main.dataset_path + "" + xlsx_filename
        xlsx = xlrd.open_workbook(xlsxFile, encoding_override='utf-8')
        sheet = xlsx.sheet_by_index(0)  # get a sheet
        flag = 2
        numberOfXlsxRows = sheet.nrows
        freeAdjValues = []
        for rowNum in range(numberOfXlsxRows):
            xlsxRow_index = rowNum + 1
            if (xlsxRow_index < maxXlsxRows + 1):
                cellValue = str(sheet.cell(xlsxRow_index - 1, 3).value).strip()
                if cellValue == "":
                    p = "null cell: " + xlsx_filename + " Row: " + str(xlsxRow_index)
                    #print(p)
                else:
                    freeAdjValues.append(cellValue)
        entr, freq = s.entropy(self, freeAdjValues)
        uniqueFreeAdjValues = list(set(freeAdjValues))
        #print(uniqueFreeAdjValues, end="\n\n")
        #print(entr, end="\n\n")
        #print(freq, end="\n\n")
        #freq.sort(key=lambda tup: tup[1])
        return uniqueFreeAdjValues, entr, freq

        # It convert the scpecific xlsx file to txt




    def entropy(self, vector):
        s = AdjectiveComparison
        key_freq = {}  # hashtable
        key_percentage_frequent_array = []
        data_entropy = 0.0
        for v in vector:
            if s.hasKey(self, key_freq, v):
                key_freq[v] += 1
            else:
                key_freq[v] = 1
        sum_of_freq = s.SumOfHashtableValues(self, key_freq)
        for freq in key_freq.values():
            data_entropy += (-freq / sum_of_freq) * np.math.log2(freq / sum_of_freq)
        for k in key_freq.keys():
            key_percentage = key_freq[k] / sum_of_freq  #*100
            key_percentage_frequent_array.append((k, round(key_percentage, 5), key_freq[k]))
        ########################
        # print(key_percentage_frequent_array)
        key_percentage_frequent_array.sort(key=lambda tup: -tup[1])
        return round(data_entropy, 2), key_percentage_frequent_array



    def countOfHashtableKeys(self, hashTable):
        count = 0
        for k in hashTable.keys():
            count += 1
        return count

    def SumOfHashtableValues(self, hashTable):
        sum = 0
        for k in hashTable.keys():
            sum += hashTable[k]
        return sum

    def hasKey(self, hashTable, key):
        for k in hashTable.keys():
            if k == key:
                return 1
        return 0


AdjectiveComparison()
