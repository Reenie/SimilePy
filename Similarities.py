import numpy as np
import main as main
import VectorSpace_simile as VSS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score

class Similarities:

    def __init__(self):
        s = Similarities
        #s.compute_centroid_Vectors(self)
        #s.cosineSimilarity(self)
        #s.print_CosSimilarity(self)
        #s.print_jacardSimilarity(self)
        #s.sameVectorsWithOtherSimiles(self)
        s.print_theSameVectorsPerSimile(self)


    def print_CosSimilarity(self):
        s = Similarities
        cs = s.cosineSimilarity(self)
        heading = ("",)
        for i in range(1, 21):
            heading = heading + (i,)
        #print(heading)
        print('\r%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s' % (heading))
        for v in cs:
            t = tuple(v[1])
            print('\r%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s' % ((v[0],) + t))

    def print_theSameVectorsPerSimile(self):
            s = Similarities
            sv = s.theSameVectorsWithOtherSimiles(self)
            print("For each simile, the number and percentage of the same vectors of other similes:")
            for v in sv:
                print(v)
                print('\n')


    def print_jacardSimilarity(self):
        s = Similarities
        js = s.jacard_similarity(self)
        heading = ("",)
        for i in range(1, 21):
            heading = heading + (i,)
        #print(heading)
        print('\r%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s' % (heading))
        for v in js:
            t = tuple(v[1])
            print('\r%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s%-7s' % ((v[0],) + t))



    def compute_centroid_Vectors(self):
        meanVector_per_simle = []
        sdVector_per_simle = []
        #for filename in main.filenames:
        numericalFeature_names, numericalVS = VSS.VectorSpace_simile.numericalVectorSpace(self, main.filenames)
        #print(numericalFeature_names)
        #print(numericalVS)
        a = np.array(numericalVS).astype(np.float)
        #b = a[0:, 3:]
        for i in range(1,21):
            #print(i)
            b = []
            for v in a:
                if v[0] == i:
                    b.append(v[3:])
            mean_row = np.mean(b, axis=0)
            sd_row = np.std(b, axis=0)
            meanVector_per_simle.append((i, mean_row))
            sdVector_per_simle.append((i, sd_row))
        #for v in meanVector_per_simle:
        #    print(v)
        return meanVector_per_simle, sdVector_per_simle

    def cosineSimilarity(self):
        s = Similarities
        cos_similarities = []
        meanVector, sdVector = s.compute_centroid_Vectors(self)
        for v in meanVector:
            row_temp = []
            for vv in meanVector:
                #print(v[1])
                #print(v[1].reshape(1, -1))
                c = cosine_similarity(v[1].reshape(1, -1), vv[1].reshape(1, -1))
                #jacard_sim = jaccard_similarity_score(v[1].reshape(1, -1), y_pred)
                #print(round(c[0][0],3))
                row_temp.append(round(c[0][0], 3))
            cos_similarities.append((v[0], row_temp))
        #print(similarities)
        return cos_similarities

    #for each simile, it returns the number of the same vectors of the other similes
    def theSameVectorsWithOtherSimiles(self):
        s = Similarities
        numericalFeature_names, numericalVS = VSS.VectorSpace_simile.numericalVectorSpace(self, main.filenames)
        theSame_vectors = []
        for i in range(1, 21):
            #print(main.filenames[i-1])
            a = []
            for v in numericalVS:
                #print(v)
                #print(v[0])
                if v[0] == str(i):
                    a.append(v[3:])
            row_temp = []
            for k in range(1, 21):
                b = []
                for v in numericalVS:
                    if v[0] == str(k):
                        b.append(v[3:])
                count_commonElements = 0
                for v1 in a:
                    if v1 in b:
                        count_commonElements = count_commonElements + 1
                temp_tuple = (k, count_commonElements, str(round(count_commonElements*100/len(a), 1))+"%")

                row_temp.append(temp_tuple)
            row_temp.sort(key=lambda x: -x[1])
            theSame_vectors.append((main.filenames[i-1], row_temp))
        #print(theSame_vectors)
        return theSame_vectors


    def jacard_similarity(self):
        s = Similarities
        size = len(main.filenames)
        numericalFeature_names, numericalVS = VSS.VectorSpace_simile.numericalVectorSpace(self, main.filenames)
        #vectorSpace = np.array(numericalVS)
        jacard_sim = []
        for i in range(1, 21):
            print(main.filenames[i-1])
            a = []
            for v in numericalVS:
                #print(v)
                #print(v[0])
                if v[0] == str(i):
                    a.append(v[3:])
            row_temp = []
            for k in range(1, 21):
                b = []
                for v in numericalVS:
                    if v[0] == str(k):
                        b.append(v[3:])
                count_commonElements = 0
                if len(a) > len(b):
                    for v1 in b:
                        if v1 in a:
                            count_commonElements = count_commonElements + 1
                else:
                    for v1 in a:
                        if v1 in b:
                            count_commonElements = count_commonElements + 1
                denominator = len(a) + len(b)- count_commonElements
                jacard_sim_temp = round(count_commonElements /(denominator + 0.0000001), 3)
                #
                string = str(count_commonElements) +"/( " + str(len(a))+ " + " + str(len(b)) + " - " + str(count_commonElements) + \
                         " = " + str(count_commonElements) + " / " + str(denominator) + " = " + str(jacard_sim_temp) + ""
                print(string + '\n')
                row_temp.append(round(jacard_sim_temp, 3))

            jacard_sim.append((i, row_temp))
        return jacard_sim





Similarities()