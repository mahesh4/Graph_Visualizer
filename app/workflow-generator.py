import math
import copy
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from app import ds_utils
G = nx.DiGraph()


def Rand(start, end, num):
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res


def absolute(causal_depth, causal_width, causal_edges):
    global G

    for w in range(causal_width):
        nodes_list = [str(w) + '_' + str(d) for d in range(causal_depth)]
        G.add_nodes_from(nodes_list)
        if w - 1 >= 0:
            # Linking causal-edges
            parent = np.random.permutation(causal_depth)
            for idx, l in enumerate(parent):
                parent_id = str(w - 1) + '_' + str(l)
                node_id = nodes_list[idx]
                G.add_edge(parent_id, node_id)
            # Adding the remaining causal_edges for each node
            if causal_edges - 1 > 0:
                for idx, node_id in enumerate(nodes_list):
                    parent_nodes = random.sample(range(0, causal_depth - 1), causal_edges - 1)
                    for parent_node in parent_nodes:
                        if parent_node == parent[idx]:
                            parent_id = str(w - 1) + '_' + str(parent_node+1)
                        else:
                            parent_id = str(w - 1) + '_' + str(parent_node)
                        G.add_edge(parent_id, node_id)
    return


def relative(causal_depth, causal_width, mean):
    for w in range(causal_width):
        nodes_list = [str(w) + '_' + str(d) for d in range(causal_depth)]
        G.add_nodes_from(nodes_list)
        if w - 1 >= 0:
            # Linking causal-edges
            parent = np.random.permutation(causal_depth)
            for idx, l in enumerate(parent):
                parent_id = str(w - 1) + '_' + str(l)
                node_id = nodes_list[idx]
                G.add_edge(parent_id, node_id)
            # Adding the remaining causal_edges for each node
            while True:
                causal_edges = int(np.random.normal(mean, causal_depth/3, 1))
                if 0 < causal_edges <= causal_depth:
                    break
            print('No of causal edges per node at layer ' + str(w) + ' is ' + str(causal_edges))
            if causal_edges - 1 > 0:
                for idx, node_id in enumerate(nodes_list):
                    parent_nodes = random.sample(range(0, causal_depth - 1), causal_edges - 1)
                    for parent_node in parent_nodes:
                        if parent_node == parent[idx]:
                            parent_id = str(w - 1) + '_' + str(parent_node + 1)
                        else:
                            parent_id = str(w - 1) + '_' + str(parent_node)
                        G.add_edge(parent_id, node_id)


def draw_graph_ntx():
    global G
    print(G.edges())
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, arrows=True)
    plt.show()


def translate_graph_to_ds_config(model_dict_template, simualtion_context, compatibility_settings):
    global G
    MONGO_CLIENT = ds_utils.connect_to_mongo()
    DS_CONFIG = {
        "simulation_context": simualtion_context,
        "compatibility_settings": compatibility_settings,
        "model_settings": {},
        "exploration": {}
    }
    for model_no, model in enumerate(list(G.nodes())):
        model_dict = copy.deepcopy(model_dict_template)
        model_dict["name"] = "model_" + model
        for idx, upstream_model in enumerate(list(G.predecessors(model))):
            model_dict["upstream_models"]["upstream_model_" + str(idx)] = "model_" + upstream_model
        for idx, downstream_model in enumerate(list(G.successors(model))):
            model_dict["downstream_models"]["downstream_model_" + str(idx)] = "model_" + downstream_model
        for sm_var_config in model_dict["sampled_variables"].values():
            sm_var_config["name"] = "model_" + model + sm_var_config["name"]
        DS_CONFIG["model_settings"]["model_" + str(model_no)] = model_dict

    MONGO_CLIENT["ds_config"]["collection"].insert_one(DS_CONFIG)
    print(DS_CONFIG)
    ds_utils.disconnect_from_mongo()


def main():
    causal_depth = int(input('Enter the causal-depth(Depth of the Layer): '))
    causal_width = int(input('Enter the causal-width(No of Layers): '))
    choice = int(input("Enter 1.absolute or 2.Relative: "))
    if choice == 1:
        causal_edges = int(input('Enter the no of causal-edges per node: '))
        absolute(causal_depth, causal_width, causal_edges)
        # draw_graph_ntx()
    elif choice == 2:
        mean = int(input('Enter the mean edges to genera'))
        relative(causal_depth, causal_width, mean)
    else:
        print('Invalid Choice')
    print("-------- workflow configuration ---------")

    model_dict_template = {
        "am_settings": {
            "am_gap": "ignore",
            "am_overlap": "ignore"
        },
        "name": "",
        "psm_settings": {
            "psm_provenance_sim": True,
            "psm_parametric_sim": True,
        },
        "sampled_variables": {},
        "sm_settings": {},
        "stateful": True,
        "temporal": {},
        "wm_settings": {},
        "upstream_models": {},
        "downstream_models": {}
    }
    print("------- Enter configurations per model -------")

    print("Temporal configuration")
    model_dict_template["temporal"]["input_window"] = int(input("Enter the input-window: "))
    model_dict_template["temporal"]["output_window"] = int(input("Enter the output-window: "))
    model_dict_template["temporal"]["shift_size"] = int(input("Enter the shift-size: "))

    print("Window manager configuration")
    model_dict_template["wm_settings"]["wm_fanout"] = int(input("Enter the wm_fanout: "))
    model_dict_template["wm_settings"]["candidates_threshold"] = float(input("Enter the wm_candidate threshold: "))
    wm_strategy = int(input("Enter the wm_strategy 1.least_gap 2.compatible 3.least_overlap 4.diverse: "))
    if wm_strategy == 1:
        model_dict_template["wm_settings"]["wm_strategy"] = "least_gap"
    elif wm_strategy == 2:
        model_dict_template["wm_settings"]["wm_strategy"] = "compatible"
    elif wm_strategy == 3:
        model_dict_template["wm_settings"]["wm_strategy"] = "least_overlap"
    else:
        model_dict_template["wm_settings"]["wm_strategy"] = "diverse"

    print("Sampling manager configuration")
    model_dict_template["sm_settings"]["sm_fanout"] = int(input("Enter the sm_fanout: "))
    sm_variables_size = int(input("Enter the number of sample-variables to generate per model: "))
    for i in range(sm_variables_size):
        sm_variable_dict = dict()
        sm_variable_dict["bin_size"] = float(input("Enter the step_size of the variable " + str(i) + ": "))
        sm_variable_dict["lower_bound"] = float(input("Enter the lower_bound of the variable " + str(i) + ": "))
        sm_variable_dict["upper_bound"] = float(input("Enter the upper_bound of the variable " + str(i) + ": "))
        sm_variable_dict["name"] = "_sm_var_" + str(i)
        model_dict_template["sampled_variables"]["variable_" + str(i)] = sm_variable_dict
    # End of loop

    print("Postsynchronization manager configuration")
    model_dict_template["psm_settings"]["psm_strategy"] = "none" if int(input("Enter PSM strategy 1.None 2.Cluster: ")) == 1 else "cluster"
    if model_dict_template["psm_settings"]["psm_strategy"] == "cluster":
        model_dict_template["psm_settings"]["psm_candidates"] = int(input("Enter the number of clusters to generate: "))

    simualation_context = {"temporal": {"begin": 1602010674, "end": 1602182780}}
    compatibility_settings = {"compatibility_strategy": "provenance", "parametric_mode": "union", "provenance_size": 10800}
    translate_graph_to_ds_config(model_dict_template, simualation_context, compatibility_settings)


if __name__ == '__main__':
    main()
    # absolute(causal_depth=3, causal_width=3, causal_edges=2)
    # relative(8, 4, 1)
    # draw_graph_ntx()
