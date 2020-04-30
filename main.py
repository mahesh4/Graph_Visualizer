
from app.model_graph import ModelGraphGenerator
from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.path_finder import PathFinder
from app.visualize_graph import VisualizeGraph

path_finder = PathFinder()
model_graph = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
visualize_graph = VisualizeGraph()
lp_solver = LpSolver()


def main():
    model_graph.generate_model_graph()
    print(model_graph.generate_model_graph())


if __name__ == '__main__':
    main()
