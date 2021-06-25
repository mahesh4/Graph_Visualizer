import numpy as np
import pandas as pd
import glob


def average_compute_times(file_name):
    compute_times = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0).to_numpy()
    compute_times_nan = np.zeros((compute_times.shape[0], compute_times.shape[1]))
    compute_times_freq = np.zeros(compute_times.shape[0])
    for i in range(0, compute_times.shape[0]):
        for j in range(0, compute_times.shape[1]):
            if compute_times[i][j] > 0:
                compute_times_nan[i][j] = compute_times[i][j]
                compute_times_freq[i] += 1
            else:
                compute_times_nan[i][j] = np.nan

    compute_times_mean = np.nanmean(compute_times_nan, axis=1)
    p = pd.DataFrame(compute_times_mean)
    new_file_name = "processed/" + file_name.split("/")[1]
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time")
    d = pd.DataFrame(compute_times_freq)
    d.to_excel(writer, sheet_name="compute frequency")
    writer.save()
    writer.close()


def perform_width_study(causal_depth, causal_edges):
    max_width = 7
    gp = pd.DataFrame()
    gp2 = pd.DataFrame()
    col_names = []
    for causal_width in range(1, max_width + 1):
        search_file = "metric" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
        for file_name in glob.glob('processed/*.xlsx', recursive=True):
            if file_name.split("/")[1] == search_file:
                p = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0)
                d = pd.read_excel(open(file_name, 'rb'), sheet_name='compute frequency', index_col=0)
                p.columns = [str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges)]
                col_names.append(str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges))
                gp = pd.concat([gp, p], axis=1, ignore_index=True)
                gp2 = pd.concat([gp2, d], axis=1, ignore_index=True)
                break
        # End of loop
    # End of loop
    gp.columns = col_names

    new_file_name = "study/width-" + str(causal_depth) + "-*-" + str(causal_edges) + ".xlsx"
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    gp.to_excel(writer)
    gp2.to_excel(writer, startcol=15)
    writer.save()
    writer.close()


def perform_depth_study(causal_width, causal_edges):
    max_depth = 7
    gp = pd.DataFrame()
    gp2 = pd.DataFrame()
    col_names = []
    for causal_depth in range(1, max_depth + 1):
        search_file = "metric" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
        for file_name in glob.glob('processed/*.xlsx', recursive=True):
            if file_name.split("/")[1] == search_file:
                p = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0)
                d = pd.read_excel(open(file_name, 'rb'), sheet_name='compute frequency', index_col=0)
                p.columns = [str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges)]
                col_names.append(str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges))
                gp = pd.concat([gp, p], axis=1, ignore_index=True)
                gp2 = pd.concat([gp2, d], axis=1, ignore_index=True)
                break
        # End of loop
    # End of loop
    gp.columns = col_names
    gp2.columns = col_names
    new_file_name = "study/depth-*-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    gp.to_excel(writer)
    gp2.to_excel(writer, startcol=15)
    writer.save()
    writer.close()


def perform_causal_study(causal_depth, causal_width):
    gp = pd.DataFrame()
    gp2 = pd.DataFrame()
    col_names = []
    for causal_edges in range(1, causal_depth + 1):
        search_file = "metric" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
        for file_name in glob.glob('processed/*.xlsx', recursive=True):
            if file_name.split("/")[1] == search_file:
                p = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0)
                d = pd.read_excel(open(file_name, 'rb'), sheet_name='compute frequency', index_col=0)
                p.columns = [str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges)]
                col_names.append(str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges))
                gp = pd.concat([gp, p], axis=1, ignore_index=True)
                gp2 = pd.concat([gp2, d], axis=1, ignore_index=True)
                break
        # End of loop
    # End of loop
    gp.columns = col_names
    gp2.columns = col_names
    new_file_name = "study/causal-" + str(causal_depth) + "-" + str(causal_width) + "-*" + ".xlsx"
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    gp.to_excel(writer)
    gp2.to_excel(writer, startcol=15)
    writer.save()
    writer.close()


def average_compute_times_diversity(causal_depth, causal_width, causal_edges):
    max_diversity = max(causal_depth * causal_width + 1, 9)
    gp = pd.DataFrame()
    columns = list()
    for diversity in range(1, max_diversity):
        search_file = "metric" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + "-D" + str(diversity) + ".xlsx"
        # print(search_file)
        for file_name in glob.glob('raw_result_diversity/*.xlsx', recursive=True):
            # print(file_name.split("/")[1])
            if file_name.split("/")[1] == search_file:
                compute_times = pd.read_excel(open(file_name, 'rb'), sheet_name='compute time', index_col=0).to_numpy()
                compute_times_mean = pd.DataFrame(np.mean(compute_times, axis=1))
                columns.append(diversity)
                gp = pd.concat([gp, compute_times_mean], axis=1, ignore_index=True)
                break

    gp.columns = columns
    new_file_name = "study_diversity/causal-" + str(causal_depth) + "-" + str(causal_width) + "-" + str(causal_edges) + ".xlsx"
    writer = pd.ExcelWriter(new_file_name, engine='xlsxwriter')
    gp.to_excel(writer)
    writer.save()
    writer.close()


if __name__ == '__main__':
    # for file_name in glob.glob('raw_results/*.xlsx', recursive=True):
    #     print(file_name)
    #     average_compute_times(file_name)
    # perform_width_study(2, 1)
    # perform_width_study(2, 2)
    # perform_width_study(3, 1)
    # perform_width_study(3, 2)
    # perform_width_study(3, 3)
    # perform_width_study(3, 2)
    # perform_width_study(4, 1)
    # perform_width_study(4, 2)
    # perform_width_study(4, 3)

    # perform_depth_study(2, 1)
    # perform_depth_study(2, 2)
    # perform_depth_study(2, 3)
    # perform_depth_study(2, 4)
    # perform_depth_study(3, 1)
    # perform_depth_study(3, 2)
    # perform_depth_study(3, 3)
    # perform_depth_study(4, 1)
    # perform_depth_study(4, 2)
    # perform_depth_study(4, 3)

    # perform_causal_study(2, 2)
    # perform_causal_study(2, 3)
    # perform_causal_study(2, 4)
    # perform_causal_study(2, 5)
    # perform_causal_study(3, 2)
    # perform_causal_study(3, 3)
    # perform_causal_study(3, 4)
    # perform_causal_study(4, 2)
    # perform_causal_study(4, 3)

    # average_compute_times_diversity(3, 3, 1)
    # average_compute_times_diversity(3, 3, 2)
    # average_compute_times_diversity(3, 3, 3)
    average_compute_times_diversity(4, 3, 1)
    # average_compute_times_diversity(4, 3, 2)
    # average_compute_times_diversity(4, 3, 3)





