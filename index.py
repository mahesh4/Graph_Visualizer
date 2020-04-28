from flask import Flask, request, abort
from app.model_graph import ModelGraphGenerator
from app.flow_graph_generator import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.db_connect import DBConnect
from app.path_finder import PathFinder
import json
app = Flask(__name__)
path_finder = PathFinder()
model_graph_generator = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
lp_solver = LpSolver()


@app.route('/')
def index():
    response = {
        'message': 'This service provided by the Center for Assured and Scalable Data Engineering, Arizona State '
                   'University. '
    }
    return response


@app.route('/graph/generate', methods=['POST'])
def generate_graph():
    global model_graph_generator, flow_graph_generator, lp_solver
    request_data = request.get_json()
    db_connect = DBConnect()
    model_graph_generator.connect_db(db_connect)
    flow_graph_generator.connect_db(db_connect)
    lp_solver.connect_db(db_connect)
    if request_data['flow_graph']:
        flow_graph_generator.generate_flow_graph()
        lp_solver.generate_edge_variables()
        lp_solver.generate_edge_constraints()
        lp_solver.generate_obj_fn()
        lp_solver.solve_optimal_path()

    if request_data['model_graph']:
        # generating the model graph
        model_graph_generator.generate_model_graph()
        # generating optimal paths
        model_graph_generator.generate_optimal_path(lp_solver.get_model())

    model_graph_generator.disconnect_db()
    lp_solver.disconnect_db()
    flow_graph_generator.disconnect_db()
    return "completed"


@app.route('/model_graph')
def model_graph():
    global model_graph_generator
    db_connect = DBConnect()
    model_graph_generator.connect_db(db_connect)
    model_graph_generator.generate_model_graph()
    response = model_graph_generator.get_model_graph()
    model_graph_generator.disconnect_db()
    return json.dumps(response)


@app.route('/model_graph/optimal_path')
def optimal_path():
    global model_graph_generator
    response = model_graph_generator.get_optimal_path()
    response = [str(node_id) for node_id in response]
    return json.dumps(response)
