import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter




class Charts:

    def __init__(self):
        s = Charts
        s.personDistribution(self)

    def personDistribution(self):
        y_values = [82.0, 93.8, 6.5, 5.1, 27.2, 86.5, 64.4, 27.7, 93.1, 67.6, 60.7, 95.8, 81.1, 17.2, 47.0, 36.9, 54.7, 16.2, 15.7, 19.6]
        x_values = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        plt.style.use('ggplot')
        plt.figure(num=None, figsize=(10, 4), dpi=10, facecolor='w', edgecolor='k')
        x_pos = [i for i, _ in enumerate(x_values)]
        plt.bar(x_pos, y_values, color='#3399ff', width=0.5)
        plt.ylabel('Συχνότητα χρήσης (%)')
        plt.xlabel('Παγιωμένη Παρομοίωση (ΠΠ)')
        plt.xticks(x_pos, x_values)
        #plt.figure(figsize=(10, 4))
        plt.show()




    '''
    def personDistribution(self):
         data = [graph.Bar(
             x=['1',  '2',  '3', '4', '5',  '6', '7',   '8',  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',  '20'],
             y=[82.0, 93.8, 6.5, 5.1, 27.2, 86.5, 64.4, 27.7, 93.1, 67.6, 60.7, 95.8, 81.1, 17.2, 47.0, 36.9, 54.7, 16.2, 15.7, 19.6]
         )]
         layout = graph.Layout(
             title='...',
         )

         fig = graph.Figure(data=data, layout=layout)
         pl.iplot(fig, filename='color-bar')
         #pl.iplot(data, filename="plot")
    '''



Charts()

