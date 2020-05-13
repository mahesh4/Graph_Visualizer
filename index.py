from flask import Flask, request, abort
from app.model_graph import ModelGraphGenerator
from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver
import json
app = Flask(__name__)
model_graph_generator = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
lp_solver = LpSolver()


@app.route('/get/model_graph', methods=['POST'])
def get_model_graph():
    global model_graph_generator, lp_solver

    request_data = request.get_json()
    try:
        if request_data['generate']:
            model_graph_generator.generate_model_graph()
            # generating the flow graph, and solving for optimal path in it
            flow_graph_generator.generate_flow_graph()
            lp_solver.generate_edge_variables()
            lp_solver.generate_edge_constraints()
            lp_solver.generate_obj_fn()
            lp_solver.solve_optimal_path()
        response = model_graph_generator.get_model_graph()
        return json.dumps(response)
    except Exception as e:
        abort(500, {'status': e})


@app.route('/get/model_graph/optimal_path', methods=['POST'])
def get_optimal_path():
    global model_graph_generator, lp_solver

    request_data = request.get_json()
    try:
        if request_data['generate']:
            model = lp_solver.get_model()
            model_graph_generator.generate_optimal_path(model)
        response = model_graph_generator.get_optimal_path()
        response = [str(node_id) for node_id in response]
        return json.dumps(response)
    except Exception as e:
        abort(500, {'status': e})


@app.route('/get/model_graph/timeline', methods=['POST'])
def get_timeline():
    global model_graph_generator

    request_data = request.get_json()
    try:
        response = model_graph_generator.get_timeline(request_data['node_id'])
        return json.dumps(response)
    except Exception as e:
        abort(500, {'status': e})
