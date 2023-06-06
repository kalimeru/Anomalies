import libraries
import functions
import parameters

# read data
data = pd.read_csv(file_GI, keep_default_na=False)
print(data.head())

# read hierarchy
hier = pd.read_csv(file_hierarchy)
unique_nodes = list(set(hier.loc[notNaN(hier['main_parent']),'main_parent']))

# clean up non used nodes
for node in unique_nodes:
    if data[node].isin([0.0]).all():
        data = data.drop([node], axis=1)

# Basic statistics
print(data.describe())

# Correlations
print(sns.heatmap(data.corr()))

# Normalization
scaler = StandardScaler()
scaler.fit(data)
array_scaled = scaler.transform(data)
data_scaled = pd.DataFrame(array_scaled, columns=data.columns)

# PCA: number of components to get 95% info explained
pca = PCA(random_state=random_state)
pca.fit(data_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= perc_explained) + 1

##########################################################################
################################## PCA ###################################
##########################################################################
# update number of components
pca = PCA(n_components=d)

# reduce dim
data_PCA = pca.fit_transform(data_scaled)
data_PCA = pd.DataFrame(data=data_PCA, index=data_scaled.index)

# undo dim reduction
data_PCA_inverse = pca.inverse_transform(data_PCA)
data_PCA_inverse = pd.DataFrame(data=data_PCA_inverse, index=data_scaled.index)

# calculate error
anomalyScoresPCA = anomalyScores(data, data_PCA_inverse)

# get the 0.1% top error
anomalies_PCA = anomalyScoresPCA.sort_values(ascending=False)[0:int(data.shape[0]*anomaly_perc)]

# plot normal and anomaly points. 1st and 2nd components
pca_aux = pd.DataFrame('Normal', index=data.index, columns=['Label'])
for i in list(anomalies_PCA.index):
    pca_aux.iloc[i] = 'Anomaly'

scatterPlot(data_PCA, pca_aux, 'PCA', [0,1], 'First Component', 'Second Component')

# feature importance on each component
loadings = pca.components_
n_features = pca.n_features_
feature_names = list(data_scaled.columns)
pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
pc_loadings = dict(zip(pc_list, loadings))
loadings_df = pd.DataFrame.from_dict(pc_loadings)
loadings_df['feature_names'] = feature_names
loadings_df = loadings_df.set_index('feature_names')
print(loadings_df)

# example: strongest anomaly
example_pca = data.iloc[anomalies_PCA.index[0],:]

##########################################################################
################################### GM ###################################
##########################################################################
# Number of clusters for GM
bgm = BayesianGaussianMixture(n_components=10, random_state=random_state, n_init=10)
bgm.fit(data_scaled)
np.round(bgm.weights_, 2) # returns 5 non-zero

# Gaussian Mixture
gm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=random_state, reg_covar=1e-3)
gm.fit(data)
print(gm.converged_) # returns true
print(gm.n_iter_) # returns 2

# get densities (anomaly measurement)
densities = gm.score_samples(data)

# threshold and top lowest density points
density_threshold = np.percentile(densities, anomaly_perc*100)
anomalies_gm = data[densities < density_threshold]

# example strongest anomaly
example_gm = data[densities==min(densities)]

##########################################################################
################################### IF ###################################
##########################################################################
# Isolation forest
isolation_forest = IsolationForest(n_estimators=100, contamination=anomaly_perc*100, random_state=random_state)
isolation_forest.fit(data)

# scoring path length
if_anomalies_scores = isolation_forest.decision_function(data)

# threshold and top shortest paths
if_threshold = np.percentile(if_anomalies_scores, anomaly_perc*100)
anomalies_if = data[if_anomalies_scores <= if_threshold]

# find top relevant features SHAP
explainer = shap.Explainer(isolation_forest.predict, data)
shap_values = explainer(data)
shap_values = shap.TreeExplainer(isolation_forest).shap_values(data)
shap.summary_plot(shap_values, data)

# example strongest anomaly
example_if = data[if_anomalies_scores == min(if_anomalies_scores)]

##########################################################################
################################ OC-SVM ##################################
##########################################################################
# One class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma=0.00005, nu=anomaly_perc*100) # random_state=42, NO VA
ocsvm.fit(data)

# scoring
svm_anomalies_scores = ocsvm.decision_function(data)

# threshold and top anomalies
svm_threshold = np.percentile(svm_anomalies_scores, anomaly_perc*100)
anomalies_svm = data[svm_anomalies_scores <= svm_threshold]

# example strongest anomaly
example_svm = anomalies_svm[anomalies_svm == min(anomalies_svm)]

##########################################################################
################################## ICA ###################################
##########################################################################
# Fast ICA
fastICA = FastICA(n_components=d, whiten='unit-variance', random_state=random_state)

# dim reduction
data_fastICA = fastICA.fit_transform(data_scaled)
data_fastICA = pd.DataFrame(data=data_fastICA, index=data_scaled.index)

# undo dim reduction
data_fastICA_inverse = fastICA.inverse_transform(data_fastICA)
data_fastICA_inverse = pd.DataFrame(data=data_fastICA_inverse, index=data_scaled.index)

# error
anomalyScores_ica = anomalyScores(data_scaled, data_fastICA_inverse)

# 0.1% top error points
anomalies_ica = anomalyScores_ica.sort_values(ascending=False)[0:int(data.shape[0]*anomaly_perc)]

# plot normal and anomaly points. 7th and 17th components
ica_aux = pd.DataFrame('Normal', index=data.index, columns=['Label'])
for i in list(anomalies_ica.index):
    ica_aux.iloc[i] = 'Anomaly'

scatterPlot(data_fastICA, ica_aux, 'ICA', [7,17], '8th Component', '18th Component')

# Correlation ICA components
loadings = fastICA.components_
n_features = fastICA.n_features_in_
feature_names = list(data_scaled.columns)
pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
pc_loadings = dict(zip(pc_list, loadings))
loadings_ica_df = pd.DataFrame.from_dict(pc_loadings)
loadings_ica_df['feature_names'] = feature_names
loadings_ica_df = loadings_ica_df.set_index('feature_names')
print(loadings_ica_df)

# example strongest anomaly
example_ica2 = data.iloc[anomalies_ica.index[0],:]

##########################################################################
################################## KNN ###################################
##########################################################################
# KNN scaled data
X_s = data_scaled.values
nbrs = NearestNeighbors(n_neighbors = 3)
nbrs.fit(X_s)

# distances and indexes of k-neaighbors from model outputs
distances, indexes = nbrs.kneighbors(X_s)

# plot mean of k-distances of each observation
plt.plot(distances.mean(axis =1))

# cutoff where 0.1% falls
knn_thresold = sorted(list(distances.mean(axis=1)), reverse=True)[int(data.shape[0]*0.001)]

# find 0.1% highest distance points
outlier_index = np.where(distances.mean(axis = 1) > knn_thresold)
anomalies_knn = pd.DataFrame(0, index=list(outlier_index[0]), columns=['Label'])

# example strongest anomaly
example_knn = data[distances.mean(axis = 1) == max(distances.mean(axis = 1))]

##########################################################################
################################# DBSCAN #################################
##########################################################################
# search eps
if False:
    n_anomalies_dbscan = []
    for e in [i/10 for i in range(2,11)]:
        anomalyScores_dbscan = DBSCAN(eps=e, min_samples=10).fit(data)
        anomalies_dbscan = data[anomalyScores_dbscan.labels_ == -1]
        print("eps: {}, cantidad de anomalías: {}"
             .format(e, len(anomalies_dbscan)))
        n_anomalies_dbscan.append([e,len(anomalies_dbscan)])

    # plot anomalies per parameter value
    plt.plot(*zip(*n_anomalies_dbscan))
    plt.xlabel('eps')
    plt.ylabel('Number of anomalies')
    plt.show()

# DBSCAN
anomalyScores_dbscan = DBSCAN(eps = 0.4, min_samples = 10).fit(data)

# anomalies through labels
anomalies_dbscan = data[anomalyScores_dbscan.labels_ == -1]


##########################################################################
################################# LOF ####################################
##########################################################################
# search n_neighbors
if False:
    n_anomalies_lof = []
    for n in range(2, 20):
        lof = LocalOutlierFactor(n_neighbors=n, contamination=anomaly_perc*100)
        anomalyScores_lof = lof.fit_predict(data)
        anomalies_lof = data[anomalyScores_lof == -1]
        print("Número de vecinos: {}, cantidad de anomalías: {}"
              .format(n, len(anomalies_lof)))
        n_anomalies_lof.append([n,len(anomalies_lof)])

    # plot anomalies per parameter value
    plt.plot(*zip(*n_anomalies_lof))
    plt.xlabel('n_neighbors')
    plt.ylabel('Number of anomalies')
    plt.show()

# LOF
lof = LocalOutlierFactor(n_neighbors=2, contamination=anomaly_perc*100)
anomalyScores_lof = lof.fit_predict(data)

# anomalies through labels
anomalies_lof = data[anomalyScores_lof == -1]

##########################################################################
################################# MA #####################################
##########################################################################
# Model agnostic: create container
results = pd.DataFrame(0, index=data.index, columns=['n_models'])

# add all anomalies
results = add_anomalies(results, anomalies_PCA)
results = add_anomalies(results, anomalies_knn)
results = add_anomalies(results, anomalies_gm)
results = add_anomalies(results, anomalies_if)
results = add_anomalies(results, anomalies_svm)
results = add_anomalies(results, anomalies_ica)
results = add_anomalies(results, anomalies_dbscan)
results = add_anomalies(results, anomalies_lof)

# check amount of points for each voting
results.value_counts()

# get anomalies with at least voting number of votes
anomalies_total = data[list(results['n_models'] >= voting)]
