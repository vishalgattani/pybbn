# Author: Vishal Gattani
# Created: 2024-06-07

from typing import Any, Dict, List, Optional, Tuple

import pathlib
import platform
import subprocess
import warnings

import cairosvg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from graphviz import Digraph

from pybbn_assurance.doe import (
    GoalNode,
    SuccessNode,
)
from pybbn_assurance.logger import logger

from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

warnings.simplefilter(action="ignore", category=FutureWarning)
np.seterr(invalid="ignore")
pd.set_option("display.max_rows", None)


class BBN:
    def __init__(self, n_experiments: int) -> None:
        self.bbn = Bbn()
        self.join_tree: Optional[Any] = None
        self.nodes: Dict[int, Any] = {}
        self.leaf_nodes: Dict[int, Any] = {}
        self.non_leaf_nodes: Dict[int, Any] = {}
        self.goal_node: Dict[int, Any] = {}
        self.n_experiments = n_experiments
        self.assurance_case_name = "assurance_case"
        self.assurance_case_yaml_name = f"{self.assurance_case_name}.yaml"
        self.assurance_case_svg_name = f"{self.assurance_case_name}.svg"
        self.assurance_case_yaml: Optional[str] = None
        self.assurance_case_dictionary: Dict[str, Any] = {}
        self.gsn2x_executable = self.get_platform_executable()

    def get_platform_executable(self) -> str:
        system = platform.system()
        if system == "Darwin":
            return "gsn2x-macOS"
        elif system == "Linux":
            return "gsn2x"
        else:
            logger.error(
                f"Unknown operating system: {system}. Supported platforms: (macOS, Ubuntu)"
            )
            return ""

    def evidence(self, nod: str, cat: str, val: float) -> None:
        """Sets the evidence of a particular node by its name, state and probability value.

        Args:
            nod: Node name where you need to plug the evidence
            cat: Which state should the evidence be incorporated into
            val: Probability value of evidence (set to 1 as it is evidence)
        """
        ev = (
            EvidenceBuilder()
            .with_node(self.join_tree.get_bbn_node_by_name(nod))
            .with_evidence(cat, val)
            .build()
        )
        self.join_tree.set_observation(ev)

    def reset_evidence(self) -> None:
        """Resets entire evidence of the BBN to their predefined values."""
        logger.info("Resetting evidence...")
        self.join_tree.unobserve_all()

    def print_probs(self) -> None:
        """Printing Posterior Probabilities of all nodes in the BBN."""
        if self.join_tree:
            for node in self.join_tree.get_bbn_nodes():
                potential = self.join_tree.get_bbn_potential(node)
                logger.debug(f"Node: {node.to_dict()}")
                logger.debug(f"Values: {potential}")
        else:
            logger.error("Join Tree has not been set!")

    def get_probabilities_node(self, id: int) -> Optional[pd.DataFrame]:
        """Fetches posterior probabilities of particular node by using its ID.

        Args:
            id: ID assigned to node during building BBN

        Returns:
            Pandas DataFrame or None if join tree not set.
        """
        if self.join_tree:
            for node in self.join_tree.get_bbn_nodes():
                if node.to_dict()["variable"]["id"] == id:
                    potential = self.join_tree.get_bbn_potential(node)
                    df = self.potential_to_df(self.join_tree.get_bbn_potential(node))
                    return df
        else:
            logger.error("Join Tree has not been set!")
        return None

    def potential_to_df(self, p: Any) -> pd.DataFrame:
        """Dataframe of a node with its states and their probability values.

        Args:
            p: Potential values from BBN

        Returns:
            Pandas DataFrame.
        """
        data: List[Tuple[Any, float]] = []
        for pe in p.entries:
            try:
                v = pe.entries.values()[0]
            except Exception:
                v = list(pe.entries.values())[0]
            pv = pe.value
            t = (v, pv)
            data.append(t)
        return pd.DataFrame(data, columns=["val", "p"])

    def potentials_to_dfs(self) -> List[Tuple[str, pd.DataFrame]]:
        """Returns all nodes and their state values as a list of dataframes.

        Returns:
            List of (node_name, DataFrame) tuples.
        """
        data: List[Tuple[str, pd.DataFrame]] = []
        for node in self.join_tree.get_bbn_nodes():
            name = node.variable.name
            df = self.potential_to_df(self.join_tree.get_bbn_potential(node))
            t = (name, df)
            data.append(t)
        return data

    def draw_bbn(self) -> None:
        """Prints a structure of the BBN using networkx library."""
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
        node_type: Any,
    ) -> Optional[BbnNode]:
        node = None
        try:
            node = BbnNode(
                Variable(node_type.id, node_type.name, node_type.get_cpt_states()),
                node_type.get_cpt_list(),
            )
            self.bbn.add_node(node)
            id = node_type.id
            self.nodes[id] = node_type
            if type(node_type).__name__ == GoalNode.__name__:
                self.goal_node[id] = node_type
            return node
        except Exception as e:
            logger.error(f"{e}")
        return None

    def create_edge(self, from_node: BbnNode, to_node: BbnNode) -> None:
        try:
            self.nodes[from_node.variable.id].parent.append(to_node.variable.id)
            self.nodes[to_node.variable.id].child.append(from_node.variable.id)
            self.bbn.add_edge(Edge(from_node, to_node, EdgeType.DIRECTED))
        except Exception as e:
            logger.error(f"{e}")

    def set_join_tree(self) -> None:
        self.join_tree = InferenceController.apply(self.bbn)
        self.assurance_case_yaml = self.bbn2yaml()

    def get_join_tree(self) -> Optional[Any]:
        return self.join_tree

    def get_parent(self, node_id: int) -> Any:
        """Get parent nodes of a node: In BBN, directions are reversed.

        Args:
            node_id: Node identifier

        Returns:
            Parent nodes.
        """
        return self.bbn.get_children(node_id=node_id)

    def get_children(self, node_id: int) -> Any:
        """Get children nodes of a node: In BBN, the directions are reversed.

        Args:
            node_id: Node identifier

        Returns:
            Children nodes.
        """
        return self.bbn.get_parents(id=node_id)

    def get_leaf_nodes(self) -> Dict[int, Any]:
        """Leaf nodes are basically the parents as BBN reverses the direction.

        Returns:
            Dictionary of leaf nodes.
        """
        leaf_nodes: Dict[int, Any] = {}
        for node_id, node_name in self.bbn.get_i2n().items():
            if not self.get_children(node_id):
                leaf_nodes[node_id] = self.nodes.get(node_id)
            else:
                self.non_leaf_nodes[node_id] = self.nodes.get(node_id)
        self.leaf_nodes = leaf_nodes
        return leaf_nodes

    def get_node_identifiers(self) -> Any:
        logger.debug(f"{self.bbn.get_i2n()}")
        return self.bbn.i2n()

    def get_bbn_dataframe(self) -> Optional[pd.DataFrame]:
        if self.join_tree:
            df_list = []
            d: Dict[str, List[float]] = {}
            for node_id, node_name in self.bbn.get_i2n().items():
                if self.non_leaf_nodes.get(node_id, None):
                    df = self.get_probabilities_node(node_id)
                    df.p = df.p.round(4)
                    df_list.append(df)
                    d[self.non_leaf_nodes[node_id].name] = [df.p[0], df.p[1]]
            df = pd.DataFrame(d).transpose().rename(columns={0: "True", 1: "False"})
            return df
        else:
            logger.error("Join Tree has not been set!")
            return None

    def print_nodes(self) -> None:
        for node_id, node_name in self.bbn.get_i2n().items():
            self.get_probabilities_node(node_id)

    def create_flowchart(self, yaml_data: Dict[str, Any]) -> Digraph:
        # Create a Digraph object
        graph = Digraph(comment="Flowchart", format="png", graph_attr={"rankdir": "BT"})

        # Add nodes and edges based on YAML data
        for key, value in yaml_data.items():
            logger.debug(key)
            node_shape = "box" if key.startswith("G") else "ellipse"
            graph.node(key, label=value["text"], shape=node_shape)

            # Add edges for supportedBy relationships
            for supported_by in value["supportedBy"]:
                graph.edge(supported_by, key)

        return graph

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Build the assurance-case GSN dictionary (pure logic, no side effects)."""
        yaml_dict: Dict[str, Any] = {}
        for node_id, node in self.nodes.items():
            current_node_yaml_id = (
                f"Sn{node_id}" if type(node).__name__ == SuccessNode.__name__ else f"G{node_id}"
            )
            supported_by_list = []
            for id in node.child:
                if type(self.nodes[id]).__name__ == SuccessNode.__name__:
                    supported_by_list.append(f"Sn{id}")
                else:
                    supported_by_list.append(f"G{id}")
            yaml_dict[current_node_yaml_id] = {
                "text": node.name,
                "supportedBy": supported_by_list,
            }
        return yaml_dict

    def bbn2yaml(self) -> Optional[str]:
        """Write the YAML file, render the GSN diagram, and return the YAML string."""
        yaml_dict = self.to_yaml_dict()
        yaml_output = yaml.dump(yaml_dict, default_flow_style=True)
        self.assurance_case_dictionary = yaml_dict
        self.write_yaml_and_render(yaml_dict)
        return yaml_output

    def write_yaml_and_render(self, yaml_dict: Dict[str, Any]) -> None:
        """Write the GSN YAML to disk and render the assurance case diagram."""
        yaml_path = pathlib.Path(self.assurance_case_yaml_name).resolve()
        svg_path = pathlib.Path(self.assurance_case_svg_name).resolve()

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)

        command = f"./{self.gsn2x_executable} {self.assurance_case_yaml_name}"
        subprocess.run(command, shell=True)

        assert svg_path.is_file(), f"Assurance case SVG not found at {svg_path}"
        logger.debug(f"Generated assurance case SVG: {svg_path}")

    def get_assurance_case_png(self) -> str:
        """Convert the assurance-case SVG to PNG and return the PNG path."""
        svg_path = pathlib.Path(self.assurance_case_svg_name).resolve()
        png_path = svg_path.parent / f"{self.assurance_case_name}.png"
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        return str(png_path)
