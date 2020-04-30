import gurobipy as gp
from gurobipy import GRB
from app.db_connect import DBConnect


class LpSolver(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)
        self.MODEL = gp.Model("mip1")
        self.OPTIMAL_PATHS = list()

    def get_model(self):
        return self.MODEL

    def generate_edge_variables(self):
        """Function to generate the edge variables, and assign them in the Linear programming model, MODEL"""
        try:
            self.connect_db()

            self.MODEL.reset()
            edge_index = 0
            flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
            edge_collection = flow_graph_constraint_database['edge']
            edge_list = edge_collection.find({})
            # generating the edge variables
            for edge in edge_list:
                edge_names = list()
                for vertex in edge['source']:
                    edge_names.append('e' + str(edge_index))
                    self.MODEL.addVar(vtype=GRB.BINARY, name='e' + str(edge_index))
                    edge_index = edge_index + 1
                # saving the edge names  for the edge in edge_collection
                edge_collection.update({'_id': edge['_id']}, {
                    '$set': {
                        'edge_names': edge_names
                    }
                })

            # update the model with edge variables
            self.MODEL.update()

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def generate_edge_constraints(self):
        """Function to generate the edge constraints. Two main constraints generated are AND and OR edge constraints, and
         edge-edge constraints"""
        try:
            self.connect_db()

            flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
            node_collection = flow_graph_constraint_database['node']
            edge_collection = flow_graph_constraint_database['edge']
            node_list = node_collection.find({})
            constraint_node_sink = None
            constraint_count = 0

            for node in node_list:
                """
                generating backward constraints, assumption that the node appears
                in destination only once in the entire EDGE_LIST, the relationship 
                is captured in such a way
                """
                constraint_node_bk = None
                constraint_node_bk_count = 0
                bk_edge_type = None

                edge = edge_collection.find_one({
                    'destination.node_id': node['node_id'],
                    'destination.node_type': node['node_type']
                })

                if edge is not None:
                    for edge_name in edge['edge_names']:
                        if constraint_node_bk is None:
                            constraint_node_bk = self.MODEL.getVarByName(edge_name)
                        else:
                            constraint_node_bk += self.MODEL.getVarByName(edge_name)
                        constraint_node_bk_count += 1
                        bk_edge_type = edge['edge_type']

                """
                generating forward constraints, assumption that the node that appears 
                in source are all related by an OR edge from WM to JOB, AND edge from model to WM and rest all edges
                are one-one type(AND)
                """
                constraint_node_fr = None
                constraint_node_fr_count = 0
                fk_edge_type = None
                fr_edges = edge_collection.find({
                    'source.node_id': node['node_id'],
                    'source.node_type': node['node_type']
                })
                if fr_edges is not None:
                    for edge in fr_edges:
                        index = 0
                        for source in edge['source']:
                            if node['node_id'] == source['node_id'] and node['node_type'] == source['node_type']:
                                break
                            index += index
                        if constraint_node_fr is None:
                            constraint_node_fr = self.MODEL.getVarByName(edge['edge_names'][index])
                        else:
                            constraint_node_fr += self.MODEL.getVarByName(edge['edge_names'][index])
                        constraint_node_fr_count += 1

                        if fk_edge_type is None:
                            fk_edge_type = edge['edge_type']

                # backward edges are AND type and forward edges are AND type
                if bk_edge_type == 'AND' and fk_edge_type == 'AND' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
                    self.MODEL.addConstr(
                        constraint_node_fr_count * constraint_node_bk == constraint_node_bk_count * constraint_node_fr,
                        "c" + str(constraint_count))
                    constraint_count += 1
                # backward edges are OR type and forward edges are AND type
                if bk_edge_type == 'OR' and fk_edge_type == 'AND' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
                    self.MODEL.addConstr(constraint_node_bk * constraint_node_fr_count == constraint_node_fr,
                                         "c" + str(constraint_count))
                    constraint_count += 1
                # backward edges are AND type and forward edges are OR type
                if bk_edge_type == 'AND' and fk_edge_type == 'OR' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
                    self.MODEL.addConstr(constraint_node_bk == constraint_node_bk_count * constraint_node_fr,
                                         "c" + str(constraint_count))
                    constraint_count += 1
                # backward edges are OR type and forward edges are OR type
                if bk_edge_type == 'OR' and fk_edge_type == 'OR' and constraint_node_bk_count > 0 and constraint_node_fr_count > 0:
                    self.MODEL.addConstr(constraint_node_bk == constraint_node_fr, "c" + str(constraint_count))
                    constraint_count += 1
                """
                check whether the node of type 'model' is sink i.e. no further simulation dependent
                on this model. Need to generate a additional constraint for the sinks
                """
                if node['node_type'] == 'model' and constraint_node_fr_count == 0:
                    if constraint_node_sink is None:
                        constraint_node_sink = constraint_node_bk
                    else:
                        constraint_node_sink += constraint_node_bk
            """
            Adding the sink constraints 
            """
            self.MODEL.addConstr(constraint_node_sink == 1, "c" + str(constraint_count))
            constraint_count += 1

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def generate_obj_fn(self):
        """ Generating the objective function for the optimal path"""
        try:
            self.connect_db()
            objective_fn = None
            flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
            edge_collection = flow_graph_constraint_database['edge']
            edge_list = edge_collection.find({})
            for edge in edge_list:
                for (weight, edge_name) in zip(edge['weight'], edge['edge_names']):
                    if objective_fn is None:
                        objective_fn = self.MODEL.getVarByName(edge_name) * weight
                    else:
                        objective_fn += self.MODEL.getVarByName(edge_name) * weight

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


