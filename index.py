import json
import os
from flask import Flask, request, abort
from flask_cors import CORS
from flask import g
from bson.objectid import ObjectId

# Custom imports
from app.model_graph import ModelGraph
from app.timelines import Timelines
from app.db_connect import DBConnect

app = Flask(__name__)
CORS(app)


def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'db'):
        db = DBConnect()
        db.connect_db()
        g.db = db
    return g.db.get_connection()


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.disconnect_db()


@app.route('/', methods=['GET'])
def index():
    response = {
        'message': 'This service provided by the Emit lab, Arizona State '
                   'University. '
    }
    return response


@app.route('/model_graph/', methods=['POST'])
@app.route('/get/model_graph', methods=["POST"])
def get_model_graph():
    try:
        request_data = request.get_json()
        if "workflow_id" in request_data:
            request_data = request.get_json()
            mongo_client, graph_client = get_db()
            model_graph = ModelGraph(mongo_client, graph_client, ObjectId(request_data["workflow_id"]))
            # Deleting any existing data in DB if any
            model_graph.delete_data()
            response = model_graph.generate_model_graph()
            # Generating window_num for nodes in model_graph
            model_graph.generate_window_number()
            return json.dumps(response)
        else:
            raise Exception("Invalid arguments passed")
    except Exception as e:
        abort(500, {'status': e})


@app.route("/model_graph/top_k_timelines", methods=["POST"])
@app.route("/get/model_graph/timeline", methods=["POST"])
def get_top_k_timelines():
    try:
        request_data = request.get_json()
        mongo_client, graph_client = get_db()
        timelines = Timelines(mongo_client, graph_client)
        if "number" in request_data:
            response = timelines.get_top_k_timelines(request_data["number"])
        else:
            raise Exception("Invalid arguments passed")
        return json.dumps(response)
    except Exception as e:
        abort(500, {"status": e})


@app.route("/model_graph/top_k_dsir_timelines", methods=["POST"])
def get_top_k_dsir_timelines():
    try:
        request_data = request.get_json()
        mongo_client, graph_client = get_db()
        timelines = Timelines(mongo_client, graph_client)
        if "dsir_id" in request_data and "number" in request_data:
            response = timelines.get_top_k_dsir_timelines(ObjectId(request_data["dsir_id"]), request_data["number"])
        else:
            raise Exception("Invalid arguments passed")
        return json.dumps(response)
    except Exception as e:
        abort(500, {"status": e})


@app.route("/database/", methods=["GET"])
def get_database():
    try:
        with open("config.json", "r") as fp:
            response = json.load(fp)
            fp.close()

        return json.dumps(response)

    except Exception as e:
        abort(500, {'status': e})


@app.route("/database/change", methods=["POST"])
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


@app.route("/workflows/", methods=["GET"])
def get_workflows():
    try:
        mongo_client, graph_client = get_db()
        workflow_db = mongo_client["ds_config"]["workflows"]
        workflow_list = workflow_db.find({}, {"_id": 0})
        return json.dumps(workflow_list)
    except Exception as e:
        abort(500, {'status': e})


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True)
