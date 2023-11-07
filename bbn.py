import os
import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
np.seterr(invalid='ignore')
import pandas as pd
pd.set_option('display.max_rows', None)

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from logger import logger

import pybbn
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.sampling.sampling import LogicSampler

from scipy.stats import binom

class BBN():
    def __init__(self) -> None:
        pass

    def getBinomProb(self,total_exp_runs,p):
        """
        Given number of experiments to be performed and the prior probability of success, returns the list of probabilities of successes for k trials out of total n runs

        Args:
        total_exp_runs (_type_): number of experiments
        p (_type_): probability of success

        Returns:
        _type_: list of probabilities of success for k out of n runs
        """
        return list(binom.pmf(list(range(total_exp_runs + 1)),total_exp_runs, p))

    def getCDFBinomProb(self,total_exp_runs,p):
        """Given number of experiments to be performed and the prior probability of success, returns the list of cumulative probabilities of successes for k trials out of total n runs

        Args:
            total_exp_runs (_type_): number of experiments
            p (_type_): probability of success

        Returns:
            _type_: list of cumulative probabilities of success up till k out of n runs
        """
        return list(binom.cdf(list(range(total_exp_runs + 1)),total_exp_runs, p))

    def evidence(self,join_tree,ev, nod, cat, val):
        """Sets the evidence of a particular node by its name, state and probability value

        Args:
            join_tree (_type_): Bayesian Belief Network
            ev (_type_): name of evidence
            nod (_type_): Node name where you need to plug the evidence
            cat (_type_): Which state should the evidence be incorporated into (e.g: True or False state if node has 2 states true/false)
            val (_type_): Probability value of evidence (set to 1 as it is evidence)
        """
        ev = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name(nod)) \
        .with_evidence(cat, val) \
        .build()
        join_tree.set_observation(ev)

    def resetEvidence(self,join_tree):
        """Resets entrie evidence of the BBN to their predefined values

        Args:
            join_tree (_type_): clears evidence from BBN
        """
        join_tree.unobserve_all()

    def print_probs(self,join_tree):
        """Printing Posterior Probabilities

        Args:
            join_tree (_type_): Prints out all posterior probabilities of all nodes in the BBN
        """
        for node in join_tree.get_bbn_nodes():
            potential = join_tree.get_bbn_potential(node)
            print("Node:", node.to_dict())
            print("Values:")
            print(potential)
            print('----------------')

    def print_probs_node(self,join_tree,id):
        """Fetches posterior probabilities of particular node by using its ID

        Args:
            join_tree (_type_): BBN tree
            id (_type_): ID assigned to node during building BBN

        Returns:
            _type_: Pandas Dataframe
        """
        for node in join_tree.get_bbn_nodes():
            if (node.to_dict()['variable']['id']==id):
                logger.debug(f"Node:{node.variable.name}")
                potential = join_tree.get_bbn_potential(node)
                df = self.potential_to_df(join_tree.get_bbn_potential(node))
                # display(df)
                logger.debug(f"{df}")
                return df

    def potential_to_df(self,p):
        """Dataframe of a node with its states and their probability values

        Args:
            p (_type_): Potential values from BBN

        Returns:
            _type_: Pandas Dataframe
        """
        data = []
        for pe in p.entries:
            try:
                v = pe.entries.values()[0]
            except:
                v = list(pe.entries.values())[0]
            p = pe.value
            t = (v, p)
            data.append(t)
        return pd.DataFrame(data, columns=['val', 'p'])

    def potentials_to_dfs(self,join_tree):
        """Returns all nodes and their state values as a list of dataframes

        Args:
            join_tree (_type_): BBN

        Returns:
            _type_: Pandas Dataframe
        """
        data = []
        for node in join_tree.get_bbn_nodes():
            name = node.variable.name
            df = self.potential_to_df(join_tree.get_bbn_potential(node))
            t = (name, df)
            data.append(t)
        return data

    def drawBBN(self,bbn):
        """Prints a structure of the BBN suing networkx library

        Args:
            bbn (_type_): Built BBN
        """
        n, d = bbn.to_nx_graph()
        pos = nx.spring_layout(n)
        nx.draw_spring(n,with_labels=True,labels=d)
        ax = plt.gca()
        plt.show()

BBN()