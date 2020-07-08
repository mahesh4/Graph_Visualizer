import numpy
import gurobipy as gp
from gurobipy import GRB, read
from random import randrange
from app.db_connect import DBConnect


class LpSolver(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)
        self.MODEL = gp.Model("mip1")
        self.CONSTRAINT_COUNT = 0
        self.model_dependency_list = ["hurricane", "flood"]
        self.OPTIMAL_PATH = list()
        self.PARAM_DIV_PATHS = list()
        self.DATA_DIV_PATHS = list()

    def get_model(self):
        return self.MODEL

    def save_model(self):
        self.MODEL.write('model.lp')

    def generate_edge_variables(self):
        """Function to generate the edge variables, and assign them in the Linear programming model, MODEL"""
        try:
            self.connect_db()
            # Reset model
            self.MODEL.reset()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            edge_collection = model_graph_database['edge']
            edge_list = edge_collection.find({})
            # generating the edge variables
            for edge in edge_list:
                edge_name = edge['edge_name']
                self.MODEL.addVar(vtype=GRB.BINARY, name=edge_name)

            # update the model with edge variables
            self.MODEL.update()

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def add_sink_constraint(self, constraint_node_bk, constraint_node_bk_count):
        """
        check whether the node of type 'model' is sink i.e. no further simulation dependent
        on this model. Need to generate a additional constraint for the sinks
        """
        # TODO: need to check sink constraint
        constraint_node_sink = self.MODEL.getConstrByName("sink")
        lhs = constraint_node_bk * (1 / constraint_node_bk_count)
        if constraint_node_sink is not None:
            lhs += self.MODEL.getRow(constraint_node_sink)
            self.MODEL.remove(self.MODEL.getConstrByName("sink"))
            self.MODEL.update()

        # Adding the sink constraints
        self.MODEL.addConstr(lhs, GRB.EQUAL, 1, 'sink')
        self.MODEL.update()

        return

    def add_edge_constraint(self, constraint_node_fr, constraint_node_fr_count, fr_edge_type, constraint_node_bk,
                            constraint_node_bk_count, bk_edge_type):
        """Function to add the edge_constriant to thr model"""

        # backward edges are AND type and forward edges are AND type
        if bk_edge_type == 'AND' and fr_edge_type == 'AND' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
            self.MODEL.addConstr(
                constraint_node_fr_count * constraint_node_bk == constraint_node_bk_count * constraint_node_fr,
                "c" + str(self.CONSTRAINT_COUNT))
            self.CONSTRAINT_COUNT += 1

        # backward edges are OR type and forward edges are AND type
        if bk_edge_type == 'OR' and fr_edge_type == 'AND' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
            self.MODEL.addConstr(constraint_node_bk * constraint_node_fr_count == constraint_node_fr,
                                 "c" + str(self.CONSTRAINT_COUNT))
            self.CONSTRAINT_COUNT += 1

        # backward edges are AND type and forward edges are OR type
        if bk_edge_type == 'AND' and fr_edge_type == 'OR' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
            self.MODEL.addConstr(constraint_node_bk == constraint_node_bk_count * constraint_node_fr,
                                 "c" + str(self.CONSTRAINT_COUNT))
            self.CONSTRAINT_COUNT += 1

        # backward edges are OR type and forward edges are OR type
        if bk_edge_type == 'OR' and fr_edge_type == 'OR' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
            self.MODEL.addConstr(constraint_node_bk == constraint_node_fr, "c" + str(self.CONSTRAINT_COUNT))
            self.CONSTRAINT_COUNT += 1

        # Updating the model
        self.MODEL.update()
        return

    def generate_edge_constraints(self):
        """
        Function to generate the edge constraints. Two main constraints generated are "AND" and "OR" edge constraints
         """
        start_nodes = self.find_start_nodes()
        self.generate_window_number(start_nodes)

        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            node_list = start_nodes
            visited_nodes = set()
            while len(node_list) > 0:
                node_id = node_list[0]
                node = node_collection.find_one({"node_id": node_id})
                del node_list[0]
                if node_id in visited_nodes:
                    continue

                visited_nodes.add(node_id)
                # Fetching and Categorizing the forward_edges
                backward_edges_dict = dict()
                for edge_name in node["destination"]:
                    # We are not adding source since we are assuming all the source for the node is visited, before
                    # the node is processed
                    edge = edge_collection.find_one({"edge_name": edge_name})
                    source = node_collection.find_one({"node_id": edge["source"]})

                    if source["window_num"] in backward_edges_dict and \
                            source["model_type"] in backward_edges_dict[source["window_num"]]:
                        backward_edges_dict[source["window_num"]][source["model_type"]].append(edge_name)
                    else:
                        if source["window_num"] not in backward_edges_dict:
                            backward_edges_dict[source["window_num"]] = dict()

                        if source["model_type"] not in backward_edges_dict[source["window_num"]]:
                            backward_edges_dict[source["window_num"]][source["model_type"]] = list()

                        backward_edges_dict[source["window_num"]][source["model_type"]].append(edge_name)

                # Generating the backward_edge constraints
                constraint_node_bk = None
                constraint_node_bk_count = 0
                bk_edge_type = 'AND'
                for window_num, model_edges_dict in backward_edges_dict.items():
                    for edge_list in model_edges_dict.values():
                        constraint = None
                        for edge_name in edge_list:
                            if constraint is None:
                                constraint = self.MODEL.getVarByName(edge_name)
                            else:
                                constraint += self.MODEL.getVarByName(edge_name)

                        # Adding the constraint to the model
                        if len(edge_list) > 1:
                            self.add_edge_constraint(1, 1, "OR", constraint_node_bk, len(edge_list), "OR")

                        # Adding the constraint to constraint_node_bk
                        if constraint_node_bk is None:
                            constraint_node_bk = constraint
                        else:
                            constraint_node_bk += constraint
                        constraint_node_bk_count += 1

                # End case
                if len(node["source"]) == 0:
                    # Nodes without forward_edges are sink nodes, adding the nodes to the sink_constraint.
                    if node["model_type"] == self.model_dependency_list[-1]:
                        self.add_sink_constraint(constraint_node_bk, constraint_node_bk_count)
                        continue

                # Fetching and Categorizing the forward_edges
                forward_edges_dict = dict()
                for edge_name in node["source"]:
                    edge = edge_collection.find_one({"edge_name": edge_name})
                    destination = node_collection.find_one({"node_id": edge["destination"]})

                    # Adding the destination to the node_list
                    node_list.append(destination["node_id"])

                    if destination["window_num"] in forward_edges_dict and \
                            destination["model_type"] in forward_edges_dict[destination["window_num"]]:
                        forward_edges_dict[destination["window_num"]][destination["model_type"]].append(edge_name)
                    else:
                        if destination["window_num"] not in forward_edges_dict:
                            forward_edges_dict[destination["window_num"]] = dict()

                        if destination["model_type"] not in forward_edges_dict[destination["window_num"]]:
                            forward_edges_dict[destination["window_num"]][destination["model_type"]] = list()

                        forward_edges_dict[destination["window_num"]][destination["model_type"]].append(edge_name)

                # Generating the forward_edge constraints
                constraint_node_fr = None
                constraint_node_fr_count = 0
                fr_edge_type = 'AND'
                for window_num, model_nodes_dict in forward_edges_dict.items():
                    for edge_list in model_nodes_dict.values():
                        constraint = None
                        for edge_name in edge_list:
                            if constraint is None:
                                constraint = self.MODEL.getVarByName(edge_name)
                            else:
                                constraint += self.MODEL.getVarByName(edge_name)

                        # Adding the constraint to the model
                        if len(edge_list) > 1:
                            self.add_edge_constraint(constraint, len(edge_list), "OR", 1, 1, "OR")
                        # Adding the constraint to the forward_edge constraints
                        if constraint_node_fr is None:
                            constraint_node_fr = constraint
                        else:
                            constraint_node_fr += constraint
                        constraint_node_fr_count += 1

                # Adding the edge_constraint to the model
                self.add_edge_constraint(constraint_node_fr, constraint_node_fr_count, fr_edge_type,
                                         constraint_node_bk, constraint_node_bk_count, bk_edge_type)

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def generate_obj_fn(self):
        """ Generating the objective function for the optimal path"""
        # TODO: Need to work on weights
        try:
            self.connect_db()
            objective_fn = None
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            edge_collection = model_graph_database['edge']
            edge_list = edge_collection.find({})
            for edge in edge_list:
                if objective_fn is None:
                    objective_fn = self.MODEL.getVarByName(edge['edge_name']) * randrange(10)
                else:
                    objective_fn += self.MODEL.getVarByName(edge['edge_name']) * randrange(10)

            self.MODEL.setObjective(objective_fn, GRB.MAXIMIZE)
            self.MODEL.update()

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def solve_optimal_path(self):
        """Solving for optimal path, and saving them in the global variable OPTIMAL_PATH to be used in top-2 heuristic
        paths"""
        self.MODEL.optimize()
        for v in self.MODEL.getVars():
            print('%s %g' % (v.varName, v.x))
            if v.x == 1:
                self.OPTIMAL_PATH.append(v.varName)
        print('Obj: %g' % self.MODEL.objVal)
        return

    def parameter_diversity(self):
        """Function to generate new Objective function for parameter-diversity"""
        # Resetting and initializing the model
        self.MODEL.reset()
        self.MODEL = read("/Users/magesh/Documents/Graph_Visualizer/model.lp")
        self.MODEL.update()

        path_nodes_set = self.find_path_nodes(self.OPTIMAL_PATH)
        for path in self.PARAM_DIV_PATHS:
            node_set = self.find_path_nodes(path)
            path_nodes_set.update(node_set)

        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            database = self.MONGO_CLIENT['ds_results']

            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            job_collection = database['jobs']
            dsir_collection = database["dsir"]
            node_list = list(set([node["node_id"] for node in node_collection.find({}, {"node_id": 1, "_id": 0})]) -
                             path_nodes_set)

            # Categorizing the nodes in the path_node_set
            path_nodes_dict = dict()
            for node_id in path_nodes_set:
                node = node_collection.find_one({"node_id": node_id})
                if node["window_num"] in path_nodes_dict and \
                        node["model_type"] in path_nodes_dict[node["window_num"]]:
                    path_nodes_dict[node["window_num"]][node["model_type"]].append(node_id)
                else:
                    if node["window_num"] not in path_nodes_dict:
                        path_nodes_dict[node["window_num"]] = dict()

                    if node["model_type"] not in path_nodes_dict[node["window_num"]]:
                        path_nodes_dict[node["window_num"]][node["model_type"]] = list()

                    path_nodes_dict[node["window_num"]][node["model_type"]].append(node_id)

            # Generating additional variables for the objective function
            add_obj = None
            while len(node_list) > 0:
                node_id = node_list[0]
                del node_list[0]

                node = node_collection.find_one({"node_id": node_id})
                dsir = dsir_collection.find_one({"_id": node_id})
                job = job_collection.find_one({"_id": dsir["metadata"]["job_id"]})
                job_param = numpy.asarray([val for val in job["variables"].values()])
                node_name = node["model_type"] + str(node["node_id"])[-3]
                model_nodes_list = path_nodes_dict[node["window_num"]][node["model_type"]]
                for m_node_id in model_nodes_list:
                    m_dsir = dsir_collection.find_one({"_id": m_node_id})
                    m_job = job_collection.find_one({"_id": m_dsir["metadata"]["job_id"]})
                    m_job_param = numpy.asarray([val for val in m_job["variables"].values()])

                    # Finding the parametric distance
                    distance = numpy.linalg.norm(job_param - m_job_param)

                    # Generating a node constraint
                    self.MODEL.addVar(vtype=GRB.BINARY, name=node_name)
                    self.MODEL.update()
                    if add_obj is None:
                        add_obj = distance*self.MODEL.getVarByName(node_name)
                    else:
                        add_obj += distance * self.MODEL.getVarByName(node_name)

                if add_obj is not None:
                    # TODO: Need to verify logic
                    backward_edges_dict = dict()
                    for edge_name in node["destination"]:
                        # We are not adding source since we are assuming all the source for the node is visited, before
                        # the node is processed
                        edge = edge_collection.find_one({"edge_name": edge_name})
                        source = node_collection.find_one({"node_id": edge["source"]})

                        if source["window_num"] in backward_edges_dict and \
                                source["model_type"] in backward_edges_dict[source["window_num"]]:
                            backward_edges_dict[source["window_num"]][source["model_type"]].append(edge_name)
                        else:
                            if source["window_num"] not in backward_edges_dict:
                                backward_edges_dict[source["window_num"]] = dict()

                            if source["model_type"] not in backward_edges_dict[source["window_num"]]:
                                backward_edges_dict[source["window_num"]][source["model_type"]] = list()

                            backward_edges_dict[source["window_num"]][source["model_type"]].append(edge_name)

                    # Generating the backward_edge constraints
                    constraint_node_bk = None
                    constraint_node_bk_count = 0
                    bk_edge_type = 'AND'
                    for window_num, model_edges_dict in backward_edges_dict.items():
                        for edge_list in model_edges_dict.values():
                            constraint = None
                            for edge_name in edge_list:
                                if constraint is None:
                                    constraint = self.MODEL.getVarByName(edge_name)
                                else:
                                    constraint += self.MODEL.getVarByName(edge_name)

                            # Adding the constraint to constraint_node_bk
                            if constraint_node_bk is None:
                                constraint_node_bk = constraint
                            else:
                                constraint_node_bk += constraint
                            constraint_node_bk_count += 1

                    # Generating the forward_node constraint
                    constraint_node_fr = self.MODEL.getVarByName(node_name)
                    constraint_node_fr_count = 1
                    fr_edge_type = "AND"

                    self.add_edge_constraint(constraint_node_fr, constraint_node_fr_count, fr_edge_type,
                                             constraint_node_bk, constraint_node_bk_count, bk_edge_type)

            # Updating the Objective function
            objective_fn = self.MODEL.getObjective() + add_obj
            self.MODEL.setObjective(objective_fn, GRB.MAXIMIZE)
            self.MODEL.update()

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

    def find_start_nodes(self):
        """Function to find the star nodes. The start nodes don't have backlink to the same model.
        Note: The nodes can be of node_type "model" or "intermediate" """
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            nodes = node_collection.find({"destination": {"$exists": True, "$size": 0}})
            start_nodes = list()

            for model in self.model_dependency_list:
                for node in nodes:
                    if node['model_type'] == model:
                        start_nodes.append(node["node_id"])

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return start_nodes

    def generate_window_number(self, start_nodes):
        """
        Function to generate the window_num for the nodes in the graph
        :parameter
            - `start_nodes`: Nodes of each model belonging to the first-window
        """
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            window = 1
            while len(start_nodes) > 0:
                node_list = []

                for node_id in start_nodes:
                    node = node_collection.find_one({'node_id': node_id})
                    # updating the window_num and node_num for the node
                    node_collection.update({'node_id': node_id}, {'$set': {'window_num': window}})

                    # pre-processing the node
                    # finding the back-links which are connected to intermediate nodes of the same model-type
                    for edge_name in node['destination']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        source = node_collection.find_one({'node_id': edge['source']})
                        if source['model_type'] == node['model_type'] and source['node_type'] == 'intermediate':
                            # updating the window_num for the intermediate node
                            node_collection.update({'node_id': edge['source']}, {'$set': {'window_num': window}})

                    # Performing DFS to find the nodes for the next window
                    for edge_name in node['source']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        destination = node_collection.find_one({'node_id': edge['destination']})
                        if destination['model_type'] == node['model_type']:
                            if destination['node_type'] == 'model':
                                node_list.append(destination['node_id'])
                            else:
                                # Updating the window_no for the nodes of node_type 'intermediate'
                                # Move forward to the node of node_type 'model'
                                for destination_edge_name in destination['source']:
                                    destination_edge = edge_collection.find_one({'edge_name': destination_edge_name})
                                    node_list.append(destination_edge['destination'])

                start_nodes = node_list
                window += 1

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

    def find_path_nodes(self, path):
        """Function to return all the node_id in the given paths"""
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            edge_collection = model_graph_database['edge']
            path_node_set = set()
            for edge_name in path:
                edge = edge_collection.find_one({"edge_name": edge_name})
                path_node_set.add(edge["source"])
                path_node_set.add(edge["destination"])

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return path_node_set

    def generate_optimal_path(self):
        """Function to generate the optimal path for model graph by transpiling the optimal path in the flow graph.
        Its done by taking into consideration of the nodes of type 'model' which are part of the optimal path. These nodes
        are of data-type DSAR, in which the DSIRs wrap around, are the nodes in the model graph"""
        self.generate_edge_variables()
        self.generate_edge_constraints()
        self.generate_obj_fn()
        self.solve_optimal_path()
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            path = list()
            for edge_name in self.OPTIMAL_PATH:
                edge = edge_collection.find_one({"edge_name": edge_name})
                source = node_collection.find_one({"node_id": edge["source"]})
                path.append(dict({
                    "name": source["model_type"],
                    "_id": str(edge["source"]),
                    "destination": list(str(edge["destination"]))
                }))

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return path
