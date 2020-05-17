from app.model_graph import ModelGraphGenerator
from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.path import PathFinder
from app.visualize_graph import VisualizeGraph
from bson.objectid import ObjectId

from matplotlib import pyplot as plt

path_finder = PathFinder()
model_graph = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
visualize_graph = VisualizeGraph()
lp_solver = LpSolver()


def main():

    # model_graph.delete_data()
    # model_graph.generate_model_graph()
    # print(model_graph.get_model_graph())
    # lp_solver.generate_edge_variables()
    # lp_solver.generate_edge_constraints()
    # lp_solver.save_model()
    # lp_solver.generate_obj_fn()
    # lp_solver.solve_optimal_path()
    timelines = model_graph.get_timeline(ObjectId('5ebf94ab515de4366243274e'))
    print(timelines)


if __name__ == '__main__':
    main()
