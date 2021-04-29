from app.model_graph import ModelGraph
from app.timelines import Timelines
from app.db_connect import DBConnect
from bson.objectid import ObjectId


def main():
    db_connect = DBConnect()
    db_connect.connect_db()
    mongo_client, graph_client = db_connect.get_connection()
    # model_graph = ModelGraph(mongo_client, graph_client, ObjectId("5ff798c28d7eeade489771ba"))
    # model_graph.delete_graph()
    # print(model_graph.generate_model_graph())
    # model_graph.generate_window_number()

    timeline = Timelines(mongo_client, graph_client, ObjectId("5ff798c28d7eeade489771ba"))
    # timeline.get_top_k_timelines_test(10)
    timelines = timeline.get_top_k_timelines(4)
    print(timelines)
    # print("\n")
    # print( imelines[1])
    # print(timeline.format_timelines_to_output([]))
    db_connect.disconnect_db()


if __name__ == '__main__':
    main()
