from app.model_graph import ModelGraphGenerator
from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.path_2 import PathFinder
from app.visualize_graph import VisualizeGraph
from bson.objectid import ObjectId

from matplotlib import pyplot as plt

path_finder = PathFinder()
model_graph = ModelGraphGenerator()
# flow_graph_generator = FlowGraphGenerator()
visualize_graph = VisualizeGraph()
# lp_solver = LpSolver()


def main():

    model_graph.delete_data()
    model_graph.generate_model_graph()
    # print(model_graph.get_model_graph())
    # start_nodes = path_finder.find_start_nodes()
    # print(start_nodes)
    # path_finder.generate_window_number(start_nodes)
    # timelines = path_finder.get_timeline()
    # print(timelines)


if __name__ == '__main__':
    main()
