import numpy as np
import main as main
import VectorSpace_simile as VSS
from sklearn.metrics.pairwise import cosine_similarity

class Similarities:

    def __init__(self):
        s = Similarities
        #s.compute_centroid_Vectors(self)
        #s.cosineSimilarity(self)
        s.print_CosSimilarity(self)


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
        similarities = []
        meanVector = s.compute_centroid_Vectors(self)
        for v in meanVector:
            row_temp = []
            for vv in meanVector:
                #print(v[1])
                #print(v[1].reshape(1, -1))
                c = cosine_similarity(v[1].reshape(1, -1), vv[1].reshape(1, -1))
                #print(round(c[0][0],3))
                row_temp.append(round(c[0][0], 3))
            similarities.append((v[0], row_temp))
        #print(similarities)
        return similarities



Similarities()