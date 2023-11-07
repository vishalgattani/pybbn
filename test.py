from bbn import BBN

bbn = BBN()
bbn.create_bbn_node(1, "a", ["t", "f"], [1, 0, 0, 1])
bbn.set_join_tree()
bbn.print_probs()
bbn.print_probs_node(id=1)
bbn.get_children(node_id=1)
