import numpy as np
import pandas as pd
import glob


def average_compute_times(file_name):
    compute_times = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0).to_numpy()
    compute_times_nan = np.zeros((compute_times.shape[0], compute_times.shape[1]))
    for i in range(0, compute_times.shape[0]):
        for j in range(0, compute_times.shape[1]):
            if compute_times[i][j] > 0:
                compute_times_nan[i][j] = compute_times[i][j]
            else:
                compute_times_nan[i][j] = np.nan

    compute_times_mean = np.nanmean(compute_times_nan, axis=1)
    p = pd.DataFrame(compute_times_mean)
    new_file_name = "processed/" + file_name.split("/")[1]
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time")
    writer.save()
    writer.close()


def perform_width_study(causal_depth, causal_edges):
    max_width = 7
    gp = pd.DataFrame()
    col_names = []
    for causal_width in range(1, max_width + 1):
        search_file = "metric" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
        for file_name in glob.glob('processed/*.xlsx', recursive=True):
            if file_name.split("/")[1] == search_file:
                p = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0)
                p.columns = [str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges)]
                col_names.append(str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges))
                gp = pd.concat([gp, p], axis=1, ignore_index=True)
                break
        # End of loop
    # End of loop
    gp.columns = col_names
    new_file_name = "study/width-" + str(causal_depth) + "-*-" + str(causal_edges) + ".xlsx"
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    gp.to_excel(writer)
    writer.save()
    writer.close()


def perform_depth_study(causal_width, causal_edges):
    max_depth = 7
    gp = pd.DataFrame()
    col_names = []
    for causal_depth in range(1, max_depth + 1):
        search_file = "metric" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
        for file_name in glob.glob('processed/*.xlsx', recursive=True):
            if file_name.split("/")[1] == search_file:
                p = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0)
                p.columns = [str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges)]
                col_names.append(str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges))
                gp = pd.concat([gp, p], axis=1, ignore_index=True)
                break
        # End of loop
    # End of loop
    gp.columns = col_names
    new_file_name = "study/depth-*-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    gp.to_excel(writer)
    writer.save()
    writer.close()


# def perform_causal_study(causal_depth, causal_width):
#     for causal_edges in range(causal_depth):



if __name__ == '__main__':
    # for file_name in glob.glob('raw_results/*.xlsx', recursive=True):
    #     print(file_name)
    #     average_compute_times(file_name)
    # perform_width_study(2, 1)
    # perform_width_study(2, 2)
    # perform_depth_study(2, 1)
    # perform_depth_study(2, 2)
    # perform_depth_study(3, 1)
    # perform_depth_study(3, 2)
    # perform_depth_study(3, 3)
    perform_causal_study(4, 3)
    perform_causal_study(3, 3)

