import os
import pathlib
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np

np.seterr(invalid="ignore")
import pandas as pd

pd.set_option("display.max_rows", None)

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Slider

import pybbn
from doe import GoalNode, MaxThresholdNode, MinThresholdNode, SuccessNode, ThresholdNode
from logger import logger
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.sampling.sampling import LogicSampler


class BBN:
    def __init__(self) -> None:
        self.bbn = Bbn()
        self.join_tree = None
        self.nodes = {}

    def evidence(self, nod, cat, val):
        """Sets the evidence of a particular node by its name, state and probability value

        Args:
            join_tree (_type_): Bayesian Belief Network
            ev (_type_): name of evidence
            nod (_type_): Node name where you need to plug the evidence
            cat (_type_): Which state should the evidence be incorporated into (e.g: True or False state if node has 2 states true/false)
            val (_type_): Probability value of evidence (set to 1 as it is evidence)
        """
        ev = (
            EvidenceBuilder()
            .with_node(self.join_tree.get_bbn_node_by_name(nod))
            .with_evidence(cat, val)
            .build()
        )
        self.join_tree.set_observation(ev)

    def reset_evidence(self):
        """Resets entrie evidence of the BBN to their predefined values

        Args:
            join_tree (_type_): clears evidence from BBN
        """
        logger.info(f"Resetting evidence...")
        self.join_tree.unobserve_all()

    def print_probs(self):
        """Printing Posterior Probabilities

        Args:
            join_tree (_type_): Prints out all posterior probabilities of all nodes in the BBN
        """
        if self.join_tree:
            for node in self.join_tree.get_bbn_nodes():
                potential = self.join_tree.get_bbn_potential(node)
                logger.debug(f"Node: {node.to_dict()}")
                logger.debug(f"Values: {potential}")
        else:
            logger.error(f"Join Tree has not been set!")

    def print_probs_node(self, id):
        """Fetches posterior probabilities of particular node by using its ID

        Args:
            join_tree (_type_): BBN tree
            id (_type_): ID assigned to node during building BBN

        Returns:
            _type_: Pandas Dataframe
        """
        if self.join_tree:
            for node in self.join_tree.get_bbn_nodes():
                if node.to_dict()["variable"]["id"] == id:
                    logger.debug(f"Node {id}:{node.variable.name}")
                    potential = self.join_tree.get_bbn_potential(node)
                    df = self.potential_to_df(self.join_tree.get_bbn_potential(node))
                    logger.debug(f"{df}")
                    return df
        else:
            logger.error(f"Join Tree has not been set!")

    def potential_to_df(self, p):
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
        return pd.DataFrame(data, columns=["val", "p"])

    def potentials_to_dfs(self):
        """Returns all nodes and their state values as a list of dataframes

        Args:
            join_tree (_type_): BBN

        Returns:
            _type_: Pandas Dataframe
        """
        data = []
        for node in self.join_tree.get_bbn_nodes():
            name = node.variable.name
            df = self.potential_to_df(self.join_tree.get_bbn_potential(node))
            t = (name, df)
            data.append(t)
        return data

    def draw_bbn(self):
        """Prints a structure of the BBN suing networkx library

        Args:
            bbn (_type_): Built BBN
        """
        try:
            n, d = self.bbn.to_nx_graph()
            logger.debug(d)
            d = {key: key for key, value in d.items()}
            pos = nx.spring_layout(n)
            nx.draw_spring(n, with_labels=True, labels=d)
            ax = plt.gca()
            plt.show()
        except Exception as e:
            logger.error(f"{e}")

    def create_bbn_node(
        self,
        id,
        name,
        node_type: [ThresholdNode, MaxThresholdNode, MinThresholdNode, GoalNode],
    ):
        node = None
        try:
            logger.debug(f"{id} CPT STATES: {(node_type.get_cpt_states())}")
            logger.debug(f"{id} CPT LIST: {node_type.get_cpt_list()}")
            node = BbnNode(
                Variable(id, name, node_type.get_cpt_states()), node_type.get_cpt_list()
            )
            self.bbn.add_node(node)
            self.nodes[id] = node_type
            logger.debug(f"Added {name}({id}) of type '{type(node_type).__name__}'")
            return node
        except Exception as e:
            logger.error(f"{e}")
        return None

    def create_edge(self, from_node, to_node):
        try:
            self.bbn.add_edge(Edge(from_node, to_node, EdgeType.DIRECTED))
        except Exception as e:
            logger.error(f"{e}")

    def set_join_tree(self):
        self.join_tree = InferenceController.apply(self.bbn)

    def get_join_tree(self):
        return self.join_tree

    def get_parent(self, node_id):
        """Get parent nodes of a node: In BBN, directions are reversed.

        Args:
            node_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        # logger.debug(f"{self.bbn.get_children(node_id=node_id)}")
        return self.bbn.get_children(node_id=node_id)

    def get_children(self, node_id):
        """Get children nodes of a node: In BBN, the directions are reversed.

        Args:
            node_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        # logger.debug(f"{self.bbn.get_parents(id=node_id)}")
        return self.bbn.get_parents(id=node_id)

    def get_leaf_nodes(self):
        """Leaf nodes are basically the parents as BBN reverses the direction.

        Returns:
            _type_: _description_
        """
        leaf_nodes = {}
        for node_id, node_name in self.bbn.get_i2n().items():
            logger.debug(
                f"Children of {node_name}({node_id}):{self.get_children(node_id=node_id)}"
            )
            logger.debug(
                f"Parent of {node_name}({node_id}):{self.get_parent(node_id=node_id)}"
            )
            if not self.get_children(node_id):
                leaf_nodes[node_id] = self.nodes.get(node_id)
        logger.debug(f"Leaf nodes: {leaf_nodes}")
        return leaf_nodes

    def get_node_identifiers(self):
        logger.debug(f"{self.bbn.get_i2n()}")
        return self.bbn.i2n()

    def get_bbn_dataframe(self):
        if self.join_tree:
            NotImplemented
        else:
            logger.error(f"Join Tree has not been set!")

    def print_nodes(self):
        for node_id, node_name in self.bbn.get_i2n().items():
            self.print_probs_node(node_id)
