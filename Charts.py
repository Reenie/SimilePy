import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr


class Charts:
    similes  = ["1_Άσπρος σαν το πανί",
             "2_Στολισμένος σαν φρεγάτα",
             "3_Απαλός σαν πούπουλο",
             "4_Απαλός σαν χάδι",
             "5_Ελαφρύς σαν πούπουλο",
             "6_Κόκκινος σαν αστακός",
             "7_Οπλισμένος σαν αστακός",
             "8_Μαλακός σαν βούτυρο",
             "9_Γερός σαν ταύρος",
             "10_Πιστός σαν σκύλος",
             "11_Κόκκινος σαν παπαρούνα",
             "12_Ντυμένος σαν αστακός",
             "13_Κόκκινος σαν το παντζάρι",
             "14_Γλυκός σαν μέλι",
             "15_Άσπρος σαν το γάλα",
             "16_Κρύος σαν τον πάγο",
             "17_Γρήγορος σαν αστραπή",
             "18_Μαύρος σαν σκοτάδι",
             "19_Μπερδεμένος σαν το κουβάρι",
             "20_Άσπρος σαν το χιόνι"
    ]

    syntacitAndSemanticEndropy = [(1, 0.18, 0.79), (2, 0.001, 0.34), (3, 2.27, 3.23), (4, 1.9, 3.54), (5, 1.96, 2.73),
                                  (6, 0.47, 0.76), (7, 0.16, 1.46), (8, 1.26, 2.79), (9, 0.28, 0.46), (10, 1.24, 1.62),
                                  (11, 0.77, 1.9),(12, 0.5, 0.25),(13, 0.83, 0.85 ),(14, 1.44, 3.53),(15, 1.27, 2.61),
                                  (16, 1.32, 3.06),(17, 1.32, 2.32), (18, 1.88, 3.45), (19, 1.03, 3.49), (20, 1.52, 3.24)
                                  ] #,('All', 1.3, 3.1)]



    def __init__(self):
        s = Charts
        s.sematicsDistribution(self)
        #s.genderDistribution(self, percentage = 0)
        #s.simileDistribution(self)
        #s.personDistribution(self)
        #s.correlationOfSimilarityAndDiversion(self)
        #s.personDistribution(self)
        #s.syntacitcAndSemanticVariability(self)

    def genderDistribution(self, percentage = 1):
        s = Charts
        m = [217, 1,   3,   30,  38,  61,  257, 52,  75,  33,   42,   16,   62,   103,  108,  64,   147,  14,   17,   186]
        f = [158, 15,  20,  69,  65,  31,  51,  46,  19,  23,   74,   7,    42,   157,  209,  101,  67,   29,   36,   278]
        n = [87,  0,   13,  75,  98,  12,  80,  64,  8,   13,   57,   1,    18,   241,  200,  107,  61,   38,   31,   635]
        mm = []
        ff = []
        nn = []
        for a, b, c in zip(m, f, n):
            mm.append(round(100 * a / (a+b+c), 2))
            ff.append(round(100 * b / (a+b+c), 2))
            nn.append(round(100 * c / (a+b+c), 2))
        if percentage == 1:
            m = mm
            f = ff
            n = nn
        #x= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        x_values = s.similes
        plt.style.use('ggplot')
        fig = plt.figure(num=None, figsize=(10, 4), dpi=250, facecolor='w', edgecolor='k')
        x_pos = [i for i, _ in enumerate(x_values)]
        colors = ['green', 'red', 'blue','#3399ff',   '#ff4d4d', '#33cc33', '#f1a629', '#f49f9c', '#3399ff']  #
        p3 = plt.bar(range(len(n)), n, color=colors[5], width=0.5)
        p2 = plt.bar(range(len(f)), f, color=colors[4], width=0.5, bottom=n)
        p1 = plt.bar(range(len(m)), m, color=colors[3], width=0.5, bottom=np.array(n) + np.array(f))
        if percentage == 1:
            plt.ylabel('Στιγμιότυπα (%)', fontsize=13, color='black')
        else:
            plt.ylabel('Στιγμιότυπα', fontsize=13, color='black')
        plt.xlabel('Παρομοίωση', fontsize=13, color='black')
        fig.autofmt_xdate(bottom=0, rotation=55, ha='right')
        plt.xticks(x_pos, x_values, fontsize=13, color='black')
        plt.legend((p1[0], p2[0], p3[0]), ('Αρσενικό', 'Θηλυκό', 'Ουδέτερο'))
        #plt.set_xticklabels(x_values, rotation='vertical')
        fig.savefig("Figures/Gender_distribution.png", bbox_inches='tight')
        #plt.show()



    # Pie chart
    def sematicsDistribution(self):
        s = Charts
        semantics = [('PERSON', '42.0%', 2041), ('BODY', '13.0%', 632), ('ARTIFACT', '10.8%', 526), ('FOOD', '5.6%', 272), (
        'ANIMAL', '4.9%', 239), ('COGNITION', '3.1%', 149), ('SUBSTANCE', '2.7%', 130), ('ACT', '2.6%', 125), (
        'COMMUNICATION', '2.5%', 121), ('ATTRIBUTE', '2.2%', 107), ('GROUP', '1.7%', 83), ('OBJECT', '1.6%', 80), (
        'PLANT', '1.3%', 62), ('TIME', '1.2%', 59), ('STATE', '1.1%', 54), ('FEELING', '0.9%', 42), ('OTHER', '%', 140)]

        freqofSems = []
        for v in semantics:
            freqofSems.append(v[2])
        #values = [462, 16, 36, 174, 201, 104, 388, 162, 102, 69, 173, 24, 122, 501, 517, 272, 275, 81, 84, 1099]
        semsWithPercentage = []
        summ = sum(freqofSems)
        print(summ)
        index = -1
        explode = []
        for v in semantics:
            index += 1
            semsWithPercentage.append(str(v[0]) + " (" + str(round(v[2] * 100 / summ, 1)) + "%)")
            #semsWithPercentage.append(v[0])
            explode.append(0.01)
        print(semsWithPercentage)
        explode[13] = 0.03
        explode[14] = 0.05
        explode[15] = 0.09
        explode[16] = 0.06
        #explode[15] = 0.05
        #labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
        #          '19', '20']
        #explode = [0.05, 0.04, 0.155, 0.15, 0.09, 0.11, 0.15, 0.22, 0.19, 0.13, 0.11, 0.09, 0.04, 0.05, 0.05]
        labels = semsWithPercentage
        plt.style.use('ggplot')
        fig = plt.figure(num=None, figsize=(6, 9), dpi=180, facecolor='w', edgecolor='k')

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                #return '{p:.2f}% ({v:d})'.format(p=pct, v=val)
                #return '{p:.1f}%'.format(p=pct)
                #return '{v:d}'.format(v=val)
                return ''
            return my_autopct

        #print(str(make_autopct(freqofSems)))

        patches, texts, autotexts = plt.pie(freqofSems, explode=explode, labels=labels, colors=None,
                  autopct=make_autopct(freqofSems),  pctdistance=0.8, shadow=False, labeldistance=1.04,
                  startangle=-45, radius=1, counterclock=False, wedgeprops=None, textprops=None, center=(0, 0),
                  frame=False, hold=None, data=None)
        # Make the labels on the small plot easier to read.
        for t in texts:
            t.set_size(13)
        for t in autotexts:
            t.set_size(13)
        # autotexts[0].set_color('y')

        # draw circle
        centre_circle = plt.Circle((0, 0), radius=0.5,  fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.axis('equal')
        fig.savefig("Figures/pieSems.png", bbox_inches='tight')
        fig.savefig("Figures/pieSems.pdf", bbox_inches='tight')
        # plt.show()

    #Pie chart
    def simileDistribution(self):
        s = Charts
        values = [462, 16,  36,  174, 201, 104, 388, 162, 102, 69,   173,  24,   122,  501,  517,  272,  275,  81,   84,   1099]
        similesWithPercentage = []
        summ = sum(values)
        print(summ)
        index = -1
        explode = []
        for v in values:
            index += 1
            #similesWithPercentage.append(s.similes[index] + " (" + str(round(v * 100 / summ, 1)) + "%)")
            similesWithPercentage.append(str(index+1) + " (" + str(v) + ", " + str(round(v*100/summ, 1)) + "%)")
            explode.append(0.04)
        #print(similesWithPercentage)

        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        explode =[0.05, 0.04, 0.16, 0.15, 0.09, 0.11, 0.15, 0.22, 0.19, 0.13, 0.11, 0.09, 0.04, 0.05, 0.05, 0.05, 0.04, 0.14, 0.04, 0.05]
        labels = similesWithPercentage
        plt.style.use('ggplot')
        fig = plt.figure(num=None, figsize=(4, 8), dpi=150, facecolor='w', edgecolor='k')
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                #return '{p:.2f}% ({v:d})'.format(p=pct, v=val)
                #return '{v:d}'.format(v=val)
                return ''
            return my_autopct
        patches, texts, autotexts = plt.pie(values, explode=explode, labels=labels, colors=None, autopct=make_autopct(values),#'%1.1f%%',
        pctdistance=0.75, shadow=False, labeldistance=1.06, startangle=180,
        radius=1, counterclock=False, wedgeprops=None, textprops=None,
        center=(0, 0), frame=False, hold=None, data=None)
        # Make the labels on the small plot easier to read.
        for t in texts:
            t.set_size('large')
        for t in autotexts:
            t.set_size('large')
        #autotexts[0].set_color('y')

        # draw circle
        centre_circle = plt.Circle((0, 0),  radius=0.5, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.axis('equal')
        fig.savefig("Figures/pieSimiles.png", bbox_inches='tight')
        fig.savefig("Figures/pieSimiles.pdf", bbox_inches='tight')
        #plt.show()


    def personDistribution(self):
        s = Charts
        y_values = [82.0, 93.8, 6.5, 5.1, 27.2, 86.5, 64.4, 27.7, 93.1, 67.6, 60.7, 95.8, 81.1, 17.2, 47.0, 36.9, 54.7, 16.2, 15.7, 19.6]
        #x_values = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        x_values = s.similes
        plt.style.use('ggplot')
        fig = plt.figure(num=None, figsize=(10, 4), dpi=250, facecolor='w', edgecolor='k')
        x_pos = [i for i, _ in enumerate(x_values)]
        plt.bar(x_pos, y_values, color='#3399ff', width=0.6)
        plt.ylabel('Συχνότητα χρήσης (%)', fontsize=14, color='black' )
        plt.xlabel('Παρομοίωση', fontsize=13, color='black')
        fig.autofmt_xdate(bottom=0, rotation=55, ha='right')
        plt.xticks(x_pos, x_values, fontsize=13, color='black')
        #plt.set_xticklabels(x_values, rotation='vertical')
        fig.savefig("Figures/Person_distribution.png", bbox_inches='tight')
        #plt.show()








    def syntacitcAndSemanticVariability(self):
        s = Charts
        syntactic_entropoy = []#[2.93, 1.70, 4.15, 4.51, 4.48, 3.00, 3.17, 3.92, 2.39, 4.10, 3.61, 3.07, 2.97, 4.47, 4.20, 4.17, 4.43, 4.47, 4.02, 4.44, 4.51 ]
        semantic_entropy =   []#[0.79, 0.34, 3.23, 3.54, 2.73, 0.76, 1.46, 2.79, 0.46, 1.62, 1.90, 0.25, 0.85, 3.53, 2.61, 3.06, 2.32, 3.45, 3.49, 3.24, 3.10 ]
        labels =            []# ['1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', "All"]

        for v in s.syntacitAndSemanticEndropy:
            syntactic_entropoy.append(v[1])
            semantic_entropy.append(v[2])
            labels.append(str(v[0]))



        #labels = s.similes
        y = syntactic_entropoy
        x = semantic_entropy
        x = np.array(x)
        y = np.array(y)
        # Create model
        model = LinearRegression(fit_intercept=True)
        model.fit(x[:, np.newaxis], y)
        xfit = np.linspace(0, 4, 40)
        yfit = model.predict(xfit[:, np.newaxis])
        # plot
        plt.style.use('ggplot')
        fig = plt.figure(num=None, figsize=(8, 4), dpi=150, edgecolor='k', facecolor ='white')
        ax = plt.subplot(1, 1, 1)
        plt.scatter(x, y, color='blue')
        for i, txt in enumerate(labels):
            if txt == '18':
                plt.annotate(txt, (x[i] - 0.15, y[i] + 0.07), fontsize=14)
            elif txt == '14':
                plt.annotate(txt, (x[i] + 0.02, y[i] - 0.15), fontsize=14)
            elif txt == '4':
                plt.annotate(txt, (x[i] + 0.02, y[i] + 0.02), fontsize=14)
            elif txt == '1':
                plt.annotate(txt, (x[i] + 0.02, y[i] - 0.15), fontsize=14)
            elif txt == '3':
                plt.annotate(txt, (x[i] + 0.03, y[i] - 0.1), fontsize=14)
            else:
                plt.annotate(txt, (x[i] + 0.02, y[i] + 0.02), fontsize=14)
        plt.plot(xfit, yfit, color="orange")


        #ax.set_ylabel('Syntactic Diversity')
        #ax.set_xlabel('Semantic Diversity')
        #ax.set_ylabel('Συντακτική διαφοροποίηση', fontsize=13, color='black' )
        #ax.set_xlabel('Σημασιολογική διαφοροποίηση', fontsize=13, color='black')
        ax.set_ylabel('Syntactic diversity', fontsize=15, color='black')
        ax.set_xlabel('Semantic diversity', fontsize=15, color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.set_facecolor('white')
        #plt.title("Syntactic and Semantic Diversity")
        #ax.legend()
        #plt.tight_layout()
        fig.savefig("Figures/syntAndSemDiversity.png", bbox_inches='tight')
        fig.savefig("Figures/syntAndSemDiversity.pdf", bbox_inches='tight')
        # plt.show()



    #correlation of similarity between simile and free adj and syntactic diversity of similes
    def correlationOfSimilarityAndDiversion(self):
        s = Charts
        #syntactic_entropy_all = [5.37, 4.35, 5.13, 5.16,  4.5,   3.01,  5.52,  5.33,  4.63,  5.57,  6.18,  5.66,  5.39,  5.14,   7.01]
        #syntactic_entropy_3 =[2.79,  3.22,  3.45, 3.5,    2.87,  2.28,  3.87,  3.36,  2.81,  3.89,  3.55,  3.48,  4.01,  3.9,   3.52]
        semantic_entropy =   []#[0.79,  0.34,  3.23,  3.54,  2.73,  0.76,  1.46,  2.79,  0.46,  1.62,  1.90, 0.25,   0.85,  3.53,  2.61,  3.06,  2.32,  3.45,  3.49,  3.26]
        syntactic_entropy =  []#[2.93,  1.70,  4.15,  4.51,  4.48,  3.00,  3.17,  3.92,  2.39,  4.10,  3.61,  3.07,  2.97,  4.47,  4.20,  4.17,  4.43,  4.47,  4.02,  4.44]
        similarityFreq =     [0.087, 0.231, 0.674, 0.803, 0.744, 0.143, 0.99,  0.661, 0.613, 0.892, 0.284, 0.999, 0.161, 0.874, 0.268, 0.425, 0.505, 0.557, 0.791, 0.581]
        similarityBinary =   [0.458, 0.447, 0.719, 0.722, 0.754, 0.577, 0.68,  0.728, 0.417, 0.422, 0.784, 0.471, 0.471, 0.929, 0.857, 0.772, 0.648, 0.746, 0.765, 0.792]
        labels =             []#['1',   '2',   '3',   '4',   '5',   '6',   '7',    '8',  '9',   '10',  '11',  '12',  '13',  '14',  '15',  '16',  '17',  '18',  '19',  '20']
        for v in s.syntacitAndSemanticEndropy:
            syntactic_entropy.append(v[1])
            semantic_entropy.append(v[2])
            labels.append(str(v[0]))


        print("Συσχέτιση συντακτικής και σημασιολογικής διαφοροποίησης των παρομοιώσεων με την ομοιότητα μεταξύ παρομοιώσεων και ελεύθερων επιθέτων.")
        print("\nΓια τις μετρήσεις χρησιμοποιούνται τα συντακτικά χαρακτηριστικά που η συντακτική τους διαφοροποίηση παρουσιάζει την μέγιστη συσχέτιση με τη σημασιολογική διαφοροποίηση.")
        print("Συντακτικά χαρακτηριστικά:['COMP', 'EMPP', 'IXP-PUNC', 'MWO', 'IXP-N', 'IXP-CREATIVE', TOSO, CONSTR]")
        pearson_correlation = pearsonr(syntactic_entropy, similarityFreq)
        print("\nPearson_correlation(syntactic_entropy, similarity_Freq) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 6)))
        pearson_correlation = pearsonr(syntactic_entropy, similarityBinary)
        print("Pearson_correlation(syntactic_entropoy, similarity_Binary) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 6)))
        pearson_correlation = pearsonr(semantic_entropy, similarityFreq)
        print("\nPearson_correlation(semantic_entropy, similarity_Freq) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 6)))
        pearson_correlation = pearsonr(semantic_entropy, similarityBinary)
        print("Pearson_correlation(semantic_entropy, similarity_Binary) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 6)))
        '''
        print("2. Συντακτικά χαρακτηριστικά: ['GENDER', 'MWE_TYPE', 'COMP'] ")
        pearson_correlation = pearsonr(syntactic_entropy_3, similarityFreq)
        print("\nPearson_correlation(syntactic_entropy_3, similarityFreq) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 4)))
        pearson_correlation = pearsonr(syntactic_entropy_3, similarityBinary)
        print("\nPearson_correlation(syntactic_entropy_3, similarityBinary) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 4)))

        print("3.Συντακτικά χαρακτηριστικά:['GENDER', 'MWE_TYPE', 'COMP', 'EMPP', 'IXP-PUNC', 'MWO', 'IXP-N', 'IXP-CREATIVE', 'EMPM', 'AGR', 'DETERMINER', 'IWO', 'VAR', 'IXP-W', 'MOD',]")
        pearson_correlation = pearsonr(syntactic_entropy_all, similarityFreq)
        print("\nPearson_correlation(syntactic_entropy_all, similarityFreq) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 4)))
        pearson_correlation = pearsonr(syntactic_entropy_all, similarityBinary)
        print("\nPearson_correlation(syntactic_entropy_all, similarityBinary) = " + str(
            round(pearson_correlation[0], 4)) + ",  p-value = " + str(round(pearson_correlation[1], 4)))
        '''

        y = semantic_entropy
        x = similarityBinary
        x = np.array(x)
        y = np.array(y)
        model = LinearRegression(fit_intercept=True)
        model.fit(x[:, np.newaxis], y)
        xfit = np.linspace(0.3, 1, 40)
        yfit = model.predict(xfit[:, np.newaxis])
        # plot
        plt.style.use('ggplot')
        fig1 = plt.figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
        ax = plt.subplot(1, 1, 1)
        plt.scatter(x, y, color='blue')
        for i, txt in enumerate(labels):
            if txt == '18':
                plt.annotate(txt, (x[i] - 0.001, y[i] - 0.24), fontsize=15)
            elif txt == '5':
                plt.annotate(txt, (x[i] + 0.005, y[i] - 0.2), fontsize=15)
            elif txt == '16':
                 plt.annotate(txt, (x[i] + 0.005, y[i] - 0.22), fontsize=15)
            elif txt == '14':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=15)
            elif txt == '4':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=15)
            elif txt == '1':
                plt.annotate(txt, (x[i] - 0.022, y[i] + 0.06), fontsize=15)
            elif txt == '13':
                plt.annotate(txt, (x[i] + 0.005, y[i] - 0.15), fontsize=15)
            elif txt == '20':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.01), fontsize=15)
            else:
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=15)
        plt.plot(xfit, yfit, color="orange")
        ax.set_ylabel('Semantic Diversity',fontsize=16, color='black')
        ax.set_xlabel('Semantic Similarity Measure 2', fontsize=16, color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.set_facecolor('white')
        fig1.savefig("Figures/SemantDivAndFreeAdjSimilar.png", bbox_inches='tight')
        fig1.savefig("Figures/SemantDivAndFreeAdjSimilar.pdf", bbox_inches='tight')

        y = syntactic_entropy
        x = similarityBinary
        x = np.array(x)
        y = np.array(y)
        fig2 = plt.figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
        ax2 = plt.subplot(1, 1, 1)
        model = LinearRegression(fit_intercept=True)
        model.fit(x[:, np.newaxis], y)
        xfit = np.linspace(0.3, 1, 40)
        yfit = model.predict(xfit[:, np.newaxis])
        plt.scatter(x, y, color='blue')
        for i, txt in enumerate(labels):
            if txt == '18':
                plt.annotate(txt, (x[i] + 0.008, y[i] - 0.12), fontsize=15)
            elif txt == '14':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=15)
            elif txt == '3':
                plt.annotate(txt, (x[i] + 0.005, y[i] - 0.1), fontsize=15)
            elif txt == '16':
                plt.annotate(txt, (x[i] + 0.006, y[i] - 0.03), fontsize=15)
            elif txt == '8':
                plt.annotate(txt, (x[i] + 0.005, y[i] - 0.09), fontsize=15)
            elif txt == '4':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=15)
            elif txt == '20':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.01), fontsize=15)
            elif txt == '2':
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.01), fontsize=15)
            elif txt == '1':
                plt.annotate(txt, (x[i] + 0.008, y[i] + 0.005), fontsize=15)
            else:
                plt.annotate(txt, (x[i] + 0.005, y[i] + 0.005), fontsize=15)
        plt.plot(xfit, yfit, color="orange")
        ax2.set_ylabel('Syntactic Diversity', fontsize=16, color='black')
        ax2.set_xlabel('Semantic Similarity Measure 2', fontsize=16, color='black')
        #plt.title("Syntactic diversity of simile and similarity with free adj")
        ax2.tick_params(axis='x', colors='black')
        ax2.tick_params(axis='y', colors='black')
        ax2.spines['bottom'].set_color('black')
        ax2.spines['top'].set_color('black')
        ax2.spines['right'].set_color('black')
        ax2.spines['left'].set_color('black')
        ax2.set_facecolor('white')
        fig2.savefig("Figures/SyntDivAndFreeAdjSimilar.png", bbox_inches='tight')
        fig2.savefig("Figures/SyntDivAndFreeAdjSimilar.pdf", bbox_inches='tight')


        #plt.show()


Charts()

