import json
import os
from random import randrange

from bson.objectid import ObjectId
from flask import Flask, request, abort
from flask_cors import CORS

from app.flow_graph import FlowGraphGenerator
from app.lp_solver import LpSolver
from app.model_graph import ModelGraphGenerator
from app.timelines import Timelines

app = Flask(__name__)
CORS(app)
model_graph_generator = ModelGraphGenerator()
flow_graph_generator = FlowGraphGenerator()
lp_solver = LpSolver()
timelines = Timelines()


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
        model_graph_generator.delete_data()
        response = model_graph_generator.generate_model_graph()
        return json.dumps(response)
    except Exception as e:
        abort(500, {'status': e})


@app.route('/get/model_graph/optimal_path', methods=['GET'])
def get_optimal_path():
    global lp_solver

    try:
        response = lp_solver.generate_optimal_path()
        return json.dumps(response)

    except Exception as e:
        abort(500, {'status': e})


@app.route('/get/model_graph/timeline', methods=['POST'])
def get_timeline():
    global model_graph_generator

    request_data = request.get_json()
    try:
        node_id = timelines.find_node(ObjectId(request_data['node_id']))
        response = timelines.get_timeline(node_id)

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
                key_count += 1
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
