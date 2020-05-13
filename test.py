from app.model_graph import ModelGraphGenerator
from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.path_finder import PathFinder
from app.visualize_graph import VisualizeGraph
from matplotlib import pyplot as plt

path_finder = PathFinder()
model_graph = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
visualize_graph = VisualizeGraph()
lp_solver = LpSolver()


def main():
    # flow_graph_generator.delete_data()
    # flow_graph_generator.job_to_model()
    # flow_graph_generator.generate_flow_graph()
    # lp_solver.generate_edge_variables()
    # lp_solver.generate_edge_constraints()
    # visualize_graph.draw_graph_ntx()
    model_graph.generate_model_graph()
    print(model_graph.get_model_graph())


if __name__ == '__main__':
    main()
