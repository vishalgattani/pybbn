# https://towardsdatascience.com/how-to-train-a-bayesian-network-bn-using-expert-knowledge-583135d872d7

import sorobn as hh
import pandas as pd

bn = hh.BayesNet(('Cloudy', 'Sprinkler'),
                ('Cloudy', 'Rain'),
                ('Sprinkler', 'Wet Grass'),
                ('Rain', 'Wet Grass'))

bn.P['Cloudy'] = pd.Series({False: .5, True: .5})
bn.P['Rain'] = pd.Series({
    (True, True): .8,
    (True, False): .2,
    (False, True): .2,
    (False, False): .8})

bn.P['Sprinkler'] = pd.Series({
    (True, True): .5,
    (True, False): .9,
    (False, True): .5,
    (False, False): .1})

# sprinkler,rain, wet grass
bn.P['Wet Grass'] = pd.Series({
    (True, True, True): .99,
    (True, True, False): .01,
    (True, False, True): .9,
    (True, False, False): .1,
    (False, True, True): .9,
    (False, True, False): .1,
    (False, False, True): .0,
    (False, False, False): 1.0})

bn.prepare()


dot = bn.graphviz()
path = dot.render('cloudy', directory='figures', format='svg', cleanup=True)