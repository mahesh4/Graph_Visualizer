from flask import Flask, request, abort
import json, os
from flask_cors import CORS, cross_origin
from bson.objectid import ObjectId

from app.model_graph import ModelGraphGenerator
from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver

app = Flask(__name__)
CORS(app)
model_graph_generator = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
lp_solver = LpSolver()


@app.route('/', methods=['GET'])
def index():
    response = {
        'message': 'This service provided by the Emit lab, Arizona State '
                   'University. '
    }
    return response


@app.route('/get/model_graph', methods=['POST'])
def get_model_graph():
    global model_graph_generator, lp_solver

    request_data = request.get_json()
    try:
        if request_data['generate']:
            model_graph_generator.delete_data()
            model_graph_generator.generate_model_graph()
            # generating the flow graph, and solving for optimal path in it
            # flow_graph_generator.generate_flow_graph()
            # lp_solver.generate_edge_variables()
            # lp_solver.generate_edge_constraints()
            # lp_solver.generate_obj_fn()
            # lp_solver.solve_optimal_path()
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
        timelines = model_graph_generator.get_timeline(ObjectId(request_data['node_id']))
        response = list()

        for timeline in timelines:
            response.append(dict({
                "score": 1,
                "links": timeline
            }))

        return json.dumps(response)

    except Exception as e:
        abort(500, {'status': e})


@app.route('/get/database', methods=["GET"])
def get_database():
    try:
        with open("config.json", "r") as fp:
            response = json.load(fp)
            fp.close()

        return json.dumps(response)

    except Exception as e:
        abort(500, {'status': e})


@app.route('/change/database', methods=["POST"])
def change_database():
    try:
        request_json = request.get_json()
        key_count = 0
        file_path = os.path.join(os.getcwd(), "config.json")
        with open(file_path, "r") as fp:
            config = json.load(fp)
            fp.close()

        for key, value in config.items():
            if key in request_json:
                key_count += 0
                config[key] = request_json[key]

        if key_count == len(request_json):
            with open(file_path, "w") as fp:
                json.dump(config, fp)
                fp.close()
        else:
            raise Exception("Key Mismatch Error")

        response = dict({"message": "IP updates successfully"})
        return json.dumps(response)

    except Exception as e:
        abort(500, {'status': e})


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
