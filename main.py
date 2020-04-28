
from app.model_graph import ModelGraphGenerator
from app.flow_graph_generator import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.db_connect import DBConnect
from app.path_finder import PathFinder

path_finder = PathFinder()
model_graph = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
lp_solver = LpSolver()


def main():
    db_connect = DBConnect()
    model_graph.connect_db(db_connect)
    response = model_graph.get_model_graph()
    model_graph.disconnect_db()


if __name__ == '__main__':
    main()
