# param
path = "C:/Users/DANIEL.PONCEMENDOZA/OneDrive - Zurich Insurance/Data Science/"
path_out = path + "Anomaly/"
path_in = path + "Download/out/"
path_initial = path + "files/"
anomaly_perc = 0.001
random_state = 42
perc_explained = 0.95
n_clusters = 5
voting = 7

# files
file_GI = path_in + "GI_items_wo_doc_type.csv"
file_hierarchy = path_initial + "IFRS4_2022Q4_main_parent.csv"

file_anomalies_7 = path_out + "all_anomalies_7.csv"
file_anomalies_8 = path_out + "all_anomalies_8.csv"
file_example = path_out + "example"
file_used_nodes = path_out + "used_nodes.csv"
file_describe = path_out + "describe.csv"
