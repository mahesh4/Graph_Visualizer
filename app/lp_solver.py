import gurobipy as gp
from gurobipy import GRB
from random import randrange
from app.db_connect import DBConnect


class LpSolver(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)
        self.MODEL = gp.Model("mip1")
        self.CONSTRAINT_COUNT = 0
        self.OPTIMAL_PATHS = list()

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
        print(constraint_node_bk)
        if constraint_node_sink is not None:
            lhs += self.MODEL.getRow(constraint_node_sink)
            self.MODEL.remove(self.MODEL.getConstrByName("sink"))
            self.MODEL.update()

        # Adding the sink constraints
        self.MODEL.addConstr(lhs, GRB.EQUAL,  1, 'sink')
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
        Function to generate the edge constraints. Two main constraints generated are AND and OR edge constraints, and
        edge-edge constraints
         """
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            node_list = list(node_collection.find({}))

            for node in node_list:
                forward_edges = list()
                backward_edges = list()
                if 'source' in node:
                    forward_edges = list(edge_collection.find({'edge_name': {'$in': node['source']}}))
                if 'destination' in node:
                    backward_edges = list(edge_collection.find({'edge_name': {'$in': node['destination']}}))

                # Fetching the backward edge variables part of the constraint
                constraint_node_bk = None
                constraint_node_bk_count = 0
                bk_edge_type = 'AND'

                for edge in backward_edges:
                    if constraint_node_bk is None:
                        constraint_node_bk = self.MODEL.getVarByName(edge['edge_name'])
                    else:
                        constraint_node_bk += self.MODEL.getVarByName(edge['edge_name'])
                    constraint_node_bk_count += 1

                if node['node_type'] == 'intermediate':
                    # Fetching the forward edge variables part of the constraint
                    constraint_node_fr = None
                    constraint_node_fr_count = 0
                    fr_edge_type = 'AND'

                    for edge in forward_edges:
                        if constraint_node_fr is None:
                            constraint_node_fr = self.MODEL.getVarByName(edge['edge_name'])
                        else:
                            constraint_node_fr += self.MODEL.getVarByName(edge['edge_name'])
                        constraint_node_fr_count += 1

                    # Adding the edge_constraint to the model
                    self.add_edge_constraint(constraint_node_fr, constraint_node_fr_count, fr_edge_type,
                                             constraint_node_bk, constraint_node_bk_count, bk_edge_type)
                else:
                    # This node_type is of model
                    # Fetching the forward edge variables part of the constraint
                    constraint_node_fr_nw = None
                    constraint_node_fr_count_nw = 0
                    fr_edge_type_nw = None

                    constraint_node_fr_sw = None
                    constraint_node_fr_count_sw = 0
                    fr_edge_type_sw = None

                    for edge in forward_edges:
                        # Fetching the destination node to check its model_type
                        destination = node_collection.find_one({'node_id': edge['source']})
                        if destination['model_type'] == node['model_type']:
                            # The edges connecting to the nodes in the next window
                            if constraint_node_fr_nw is None:
                                constraint_node_fr_nw = self.MODEL.getVarByName(edge['edge_name'])
                            else:
                                constraint_node_fr_nw += self.MODEL.getVarByName(edge['edge_name'])
                            constraint_node_fr_count_nw += 1

                        else:
                            # The edge is connecting to a node in the same window. The current node is acting as
                            # upstream model
                            if constraint_node_fr_sw is None:
                                constraint_node_fr_sw = self.MODEL.getVarByName(edge['edge_name'])
                            else:
                                constraint_node_fr_sw += self.MODEL.getVarByName(edge['edge_name'])
                            constraint_node_fr_count_sw += 1

                        # Setting the edge_types
                        if fr_edge_type_nw is None and destination['node_type'] == 'intermediate':
                            fr_edge_type_nw = 'AND'
                        else:
                            fr_edge_type_nw = 'OR'

                        if fr_edge_type_sw is None and destination['node_type'] == 'intermediate':
                            fr_edge_type_sw = 'AND'
                        else:
                            fr_edge_type_sw = 'OR'
                        # End of loop
                    # Adding the edge_constraint to the model
                    self.add_edge_constraint(constraint_node_fr_sw, constraint_node_fr_count_sw, fr_edge_type_sw,
                                             constraint_node_bk, constraint_node_bk_count, bk_edge_type)

                    self.add_edge_constraint(constraint_node_fr_nw, constraint_node_fr_count_nw, fr_edge_type_nw,
                                             constraint_node_bk, constraint_node_bk_count, bk_edge_type)
                    # Check if the node is a sink
                    # if constraint_node_fr_count_nw == 0 and constraint_node_fr_count_sw == 0:
                        # self.add_sink_constraint(constraint_node_bk, constraint_node_bk_count)

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    # def generate_edge_constraint(self):
    #     """Function to generate the edge constraints. Two main constraints generated are AND and OR edge constraints, and
    #      edge-edge constraints"""
    #     try:
    #         self.connect_db()
    #
    #         flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
    #         node_collection = flow_graph_constraint_database['node']
    #         edge_collection = flow_graph_constraint_database['edge']
    #         node_list = node_collection.find({})
    #         constraint_node_sink = None
    #         constraint_count = 0
    #
    #         for node in node_list:
    #             """
    #             generating backward constraints, assumption that the node appears
    #             in destination only once in the entire EDGE_LIST, the relationship
    #             is captured in such a way
    #             """
    #             constraint_node_bk = None
    #             constraint_node_bk_count = 0
    #             bk_edge_type = None
    #
    #             edge = edge_collection.find_one({
    #                 'destination.node_id': node['node_id'],
    #                 'destination.node_type': node['node_type']
    #             })
    #
    #             if edge is not None:
    #                 for edge_name in edge['edge_names']:
    #                     if constraint_node_bk is None:
    #                         constraint_node_bk = self.MODEL.getVarByName(edge_name)
    #                     else:
    #                         constraint_node_bk += self.MODEL.getVarByName(edge_name)
    #                     constraint_node_bk_count += 1
    #                     bk_edge_type = edge['edge_type']
    #
    #             """
    #             generating forward constraints, assumption that the node that appears
    #             in source are all related by an OR edge from WM to JOB, AND edge from model to WM and rest all edges
    #             are one-one type(AND)
    #             """
    #             constraint_node_fr = None
    #             constraint_node_fr_count = 0
    #             fk_edge_type = None
    #             fr_edges = edge_collection.find({
    #                 'source.node_id': node['node_id'],
    #                 'source.node_type': node['node_type']
    #             })
    #             if fr_edges is not None:
    #                 for edge in fr_edges:
    #                     index = 0
    #                     for source in edge['source']:
    #                         if node['node_id'] == source['node_id'] and node['node_type'] == source['node_type']:
    #                             break
    #                         index += index
    #                     if constraint_node_fr is None:
    #                         constraint_node_fr = self.MODEL.getVarByName(edge['edge_names'][index])
    #                     else:
    #                         constraint_node_fr += self.MODEL.getVarByName(edge['edge_names'][index])
    #                     constraint_node_fr_count += 1
    #
    #                     if fk_edge_type is None:
    #                         fk_edge_type = edge['edge_type']
    #
    #             # backward edges are AND type and forward edges are AND type
    #             if bk_edge_type == 'AND' and fk_edge_type == 'AND' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
    #                 self.MODEL.addConstr(
    #                     constraint_node_fr_count * constraint_node_bk == constraint_node_bk_count * constraint_node_fr,
    #                     "c" + str(constraint_count))
    #                 constraint_count += 1
    #             # backward edges are OR type and forward edges are AND type
    #             if bk_edge_type == 'OR' and fk_edge_type == 'AND' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
    #                 self.MODEL.addConstr(constraint_node_bk * constraint_node_fr_count == constraint_node_fr,
    #                                      "c" + str(constraint_count))
    #                 constraint_count += 1
    #             # backward edges are AND type and forward edges are OR type
    #             if bk_edge_type == 'AND' and fk_edge_type == 'OR' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
    #                 self.MODEL.addConstr(constraint_node_bk == constraint_node_bk_count * constraint_node_fr,
    #                                      "c" + str(constraint_count))
    #                 constraint_count += 1
    #             # backward edges are OR type and forward edges are OR type
    #             if bk_edge_type == 'OR' and fk_edge_type == 'OR' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
    #                 self.MODEL.addConstr(constraint_node_bk == constraint_node_fr, "c" + str(constraint_count))
    #                 constraint_count += 1
    #             """
    #             check whether the node of type 'model' is sink i.e. no further simulation dependent
    #             on this model. Need to generate a additional constraint for the sinks
    #             """
    #             if node['node_type'] == 'model' and constraint_node_fr_count == 0:
    #                 if constraint_node_sink is None:
    #                     constraint_node_sink = constraint_node_bk
    #                 else:
    #                     constraint_node_sink += constraint_node_bk
    #         """
    #         Adding the sink constraints
    #         """
    #         self.MODEL.addConstr(constraint_node_sink == 1, "c" + str(constraint_count))
    #         constraint_count += 1
    #
    #     except Exception as e:
    #         raise e
    #
    #     finally:
    #         self.disconnect_db()
    #
    #     return

    def generate_obj_fn(self):
        """ Generating the objective function for the optimal path"""
        # TODO: Need to work on weights
        try:
            self.connect_db()
            objective_fn = None
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            edge_collection = model_graph_database['edge']
            edge_list = list(edge_collection.find({}))
            print(edge_list)
            for edge in edge_list:
                print(edge)
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
                self.OPTIMAL_PATHS.append(v.varName)
        print('Obj: %g' % self.MODEL.objVal)
        return
