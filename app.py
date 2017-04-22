from flask import Flask
from flask import render_template
from pymongo import MongoClient
import pandas as pd
import json
import numpy as np
from bson import json_util
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn import manifold
from sklearn.metrics import euclidean_distances

from bson.json_util import dumps
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'gamedb'
COLLECTION_NAME = 'games'



#FIELDS = {'Critic_Count': True, 'Person Number': False, 'Age': False, 'Education': False, 'Earnings': False, 'Hours': False, 'Kids': False,'Married' : False,  '_id': False,}


FIELDS = {'Name': True,	'Platform': True,	'Year_of_Release':True,	'Genre':True,	'Publisher':True,	'NA_Sales':True,	'EU_Sales':True,	'JP_Sales':True,	'Other_Sales':True,	'Global_Sales':True,	'Critic_Score':True,	'Critic_Count':True,	'User_Score':True,	'User_Count':True,	'Developer':True,	'Rating':True, '_id': False}
client = MongoClient('localhost:27017')

g_pcaInputData = []
g_mdsInput = []
g_featureComponentNames = []
g_decimatedData = []
g_odecimatedData = []


g_pcaRandomInputData = []
g_randomMdsInput = []
g_randomFeatureComponentNames = []
g_randomDecimatedData = []
g_oRandomdecimatedData = []


def getRandomData():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    games = collection.find(projection=FIELDS)
    connection.close()

    df = pd.DataFrame(list(games))
    data_clean = df.dropna()
    le = preprocessing.LabelEncoder()
    le.fit(data_clean.Developer);
    data_clean.loc[:, 'Developer'] = le.transform(data_clean.Developer)
    le.fit(data_clean.Publisher);
    data_clean.loc[:, 'Publisher'] = le.transform(data_clean.Publisher)
    le.fit(data_clean.Name);
    data_clean.loc[:, 'Name'] = le.transform(data_clean.Name)
    le.fit(data_clean.Genre);
    data_clean.loc[:, 'Genre'] = le.transform(data_clean.Genre)
    le.fit(data_clean.Platform);
    data_clean.loc[:, 'Platform'] = le.transform(data_clean.Platform)
    le.fit(data_clean.Rating);
    data_clean.loc[:, 'Rating'] = le.transform(data_clean.Rating)

    #    'Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer' ,'Rating'

    cluster = data_clean[
        ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
         'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating']]
    cluster.describe()

    clustervar = cluster.copy()
    clustervar['Year_of_Release'] = preprocessing.scale(clustervar['Year_of_Release'].astype('float64'))
    clustervar['NA_Sales'] = preprocessing.scale(clustervar['NA_Sales'].astype('float64'))
    clustervar['EU_Sales'] = preprocessing.scale(clustervar['EU_Sales'].astype('float64'))
    clustervar['JP_Sales'] = preprocessing.scale(clustervar['JP_Sales'].astype('float64'))
    clustervar['Other_Sales'] = preprocessing.scale(clustervar['Other_Sales'].astype('float64'))
    clustervar['Global_Sales'] = preprocessing.scale(clustervar['Global_Sales'].astype('float64'))
    clustervar['Critic_Score'] = preprocessing.scale(clustervar['Critic_Score'].astype('float64'))
    clustervar['Critic_Count'] = preprocessing.scale(clustervar['Critic_Count'].astype('float64'))
    clustervar['User_Score'] = preprocessing.scale(clustervar['User_Score'].astype('float64'))
    clustervar['User_Count'] = preprocessing.scale(clustervar['User_Count'].astype('float64'))

    sampled_data = []
    originalDecimatedData = []
    random_sampled_indices = np.random.randint(len(clustervar), size=int(len(clustervar) * 0.05))
    for x in range(len(random_sampled_indices)):
        sampled_data.append(clustervar.ix[random_sampled_indices[x]])
    global g_oRandomdecimatedData
    g_oRandomdecimatedData = sampled_data
    connection.close()
    return sampled_data, cluster.columns.values

def getRandomPCA():
    decimatedData, featureNames = getRandomData()
    pcaInputData = np.array(decimatedData)
    pca = PCA(n_components=16)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    pipeline.fit_transform(pcaInputData)
    plt.plot(pca.explained_variance_, '--o')
    plt.axhline(y=1, color='r')
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('Eigen Values')
    plt.show()



    import csv
    with open('C:\\Users\\mayur\\Desktop\\PCARandom_Eigen.csv', 'w') as csvfile:
        fieldnames = ['Components', 'Eigen']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(16):
            writer.writerow({'Components': x, 'Eigen': pca.explained_variance_[x]})

    global g_pcaRandomInputData
    g_pcaRandomInputData = pcaInputData

    pca = PCA(n_components=5)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    pipeline.fit_transform(pcaInputData)

    matrix = np.array(pca.components_)
    Components = matrix.transpose()
    squareLoadings = []
    for x in range(len(Components)):
        y = 0
        for componentIndex in range(pca.n_components - 1):
            y = + (Components[x][componentIndex] * Components[x][componentIndex])
        squareLoadings.append(y)

    features = featureNames
    y_pos = np.arange(len(features))
    plt.bar(y_pos, squareLoadings, align='center', alpha=0.5)
    plt.xticks(y_pos, features, rotation='vertical')
    plt.ylabel('Sum Square Loadings')
    plt.title('Scree Plot')
    plt.show()

    sortedIndices = np.argsort(squareLoadings);

    import csv
    with open('C:\\Users\\mayur\\Desktop\\PCArandom_square.csv', 'w') as csvfile:
        fieldnames = ['features', 'squareloadings']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(16):
            writer.writerow({'features': features[x], 'squareloadings': squareLoadings[x]})

    pca = PCA(n_components=2)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    decimatedDataComponents = pipeline.fit_transform(pcaInputData)

    featureComponentNames = ['Component1', 'Component2']

    dimensionReducedData = []
    rowDict = dict()
    for decimatedIndex in range(len(decimatedDataComponents)):
        rowDict = dict()
        for x in range(2):
            rowDict[featureComponentNames[x]] = decimatedDataComponents[decimatedIndex][x]
        dimensionReducedData.append(rowDict)

        # dimensionReducedData.append([row[x] for row in decimatedData])
    global g_randomMdsInput
    global g_randomFeatureComponentNames
    global g_randomDecimatedData
    g_randomMdsInput = decimatedDataComponents
    g_randomFeatureComponentNames = featureComponentNames
    g_randomDecimatedData = decimatedData
    return dimensionReducedData


def getDecimatedData():


    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    games = collection.find(projection=FIELDS)
    connection.close()

    df = pd.DataFrame(list(games))
    data_clean = df.dropna()
    le = preprocessing.LabelEncoder()
    le.fit(data_clean.Developer);
    data_clean.loc[:, 'Developer'] = le.transform(data_clean.Developer)
    le.fit(data_clean.Publisher);
    data_clean.loc[:, 'Publisher'] = le.transform(data_clean.Publisher)
    le.fit(data_clean.Name);
    data_clean.loc[:, 'Name'] = le.transform(data_clean.Name)
    le.fit(data_clean.Genre);
    data_clean.loc[:,'Genre']=le.transform(data_clean.Genre)
    le.fit(data_clean.Platform);
    data_clean.loc[:, 'Platform'] = le.transform(data_clean.Platform)
    le.fit(data_clean.Rating);
    data_clean.loc[:, 'Rating'] = le.transform(data_clean.Rating)


    cluster = data_clean[['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer' ,'Rating']]
    cluster.describe()

    clustervar = cluster.copy()
    clustervar['Year_of_Release'] = preprocessing.scale(clustervar['Year_of_Release'].astype('float64'))
    clustervar['NA_Sales'] = preprocessing.scale(clustervar['NA_Sales'].astype('float64'))
    clustervar['EU_Sales'] = preprocessing.scale(clustervar['EU_Sales'].astype('float64'))
    clustervar['JP_Sales'] = preprocessing.scale(clustervar['JP_Sales'].astype('float64'))
    clustervar['Other_Sales'] = preprocessing.scale(clustervar['Other_Sales'].astype('float64'))
    clustervar['Global_Sales'] = preprocessing.scale(clustervar['Global_Sales'].astype('float64'))
    clustervar['Critic_Score'] = preprocessing.scale(clustervar['Critic_Score'].astype('float64'))
    clustervar['Critic_Count'] = preprocessing.scale(clustervar['Critic_Count'].astype('float64'))
    clustervar['User_Score'] = preprocessing.scale(clustervar['User_Score'].astype('float64'))
    clustervar['User_Count'] = preprocessing.scale(clustervar['User_Count'].astype('float64'))

    clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=0)

    from scipy.spatial.distance import cdist
    clusters = range(1, 10)
    meandist = []

    for k in clusters:
        model = KMeans(n_clusters=k)
        model.fit(clus_train)
        clusassign = model.predict(clus_train)
        meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_,'euclidean'), axis = 1)) / clus_train.shape[0])

    #plt.plot(clusters, meandist)
    #plt.xlabel('Number of clusters')
    #plt.ylabel('Average distance')
    #plt.title('Selecting k with the Elbow Method')
    #plt.show()

    import csv
    with open('C:\\Users\\mayur\\Desktop\\K.csv', 'w') as csvfile:
        fieldnames = ['clusters','meandist']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(len(clusters)):
            writer.writerow({'clusters': x, 'meandist': meandist[x]})



    model3 = KMeans(n_clusters=3)
    model3.fit(clustervar)

    Clusts = [[] for _ in range(model3.n_clusters)]
    count = 1;
    for x in np.nditer(model3.labels_):
        Clusts[x].append(count)
        count += 1

    decimatedData = []
    originalDecimatedData = []
    for clustIndex in range (model3.n_clusters):
        for index in range(np.math.ceil(len(Clusts[clustIndex]) * 0.05)):
            decimatedData.append(clustervar.ix[Clusts[clustIndex][index]])
            originalDecimatedData.append(clustervar.ix[index])

    global g_odecimatedData
    g_odecimatedData = originalDecimatedData

    return decimatedData, cluster



@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/games/pca")
def pcaVisualize():
    dimensionReducedData = getPCA()
    json_games = json.dumps(dimensionReducedData, default=json_util.default)
    return json_games



@app.route("/games/pca/random")
def pcaVisualizeRandom():
    dimensionReducedData = getRandomPCA()
    #dimensionReducedData = getRandomPCA()
    json_games_random = json.dumps(dimensionReducedData, default=json_util.default)
    return json_games_random

def getPCA():
    decimatedData, cluster = getDecimatedData()
    pcaInputData = np.array(decimatedData)
    pca = PCA(n_components = 16)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    pipeline.fit_transform(pcaInputData)
    plt.plot(pca.explained_variance_, '--o')
    plt.axhline(y=1, color='r')
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('Eigen Values')
    plt.show()

    import csv
    with open('C:\\Users\\mayur\\Desktop\\PCA_Eigen.csv', 'w') as csvfile:
        fieldnames = ['Components', 'Eigen']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(16):
            writer.writerow({'Components': x, 'Eigen': pca.explained_variance_[x]})

    global g_pcaInputData
    g_pcaInputData = pcaInputData

    pca = PCA(n_components=5)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    pipeline.fit_transform(pcaInputData)

    matrix = np.array(pca.components_)
    Components = matrix.transpose()
    squareLoadings = []
    for x in range(len(Components)):
        y = 0
        for componentIndex in range(pca.n_components - 1):
            y =+ (Components[x][componentIndex] * Components[x][componentIndex])
        squareLoadings.append(y)

    featureNames = []
    features = list(cluster.columns.values)

    y_pos = np.arange(len(features))
    plt.bar(y_pos, squareLoadings, align='center', alpha=0.5)
    plt.xticks(y_pos, features, rotation='vertical')
    plt.ylabel('Sum Square Loadings')
    plt.title('Scree Plot')
    plt.show()

    sortedIndices = np.argsort(squareLoadings);
    import csv
    with open('C:\\Users\\mayur\\Desktop\\PCA_square.csv', 'w') as csvfile:
        fieldnames = ['features', 'squareloadings']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(16):
            writer.writerow({'features': features[x], 'squareloadings': squareLoadings[x]})


    for x in range(2):
        featureNames.append(features[sortedIndices[x]])

    pca = PCA(n_components=2)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    decimatedDataComponents = pipeline.fit_transform(pcaInputData)

    featureComponentNames = ['Component1','Component2' ]

    dimensionReducedData = []
    rowDict = dict()
    for decimatedIndex in range(len(decimatedDataComponents)):
        rowDict = dict()
        for x in range(2):
            rowDict[featureComponentNames[x]] = decimatedDataComponents[decimatedIndex][x]
        dimensionReducedData.append(rowDict)
    global g_mdsInput
    global g_featureComponentNames
    global g_decimatedData
    g_mdsInput = decimatedDataComponents
    g_featureComponentNames = featureComponentNames
    g_decimatedData = decimatedData
    return dimensionReducedData


@app.route("/games/mdsEuclidean")
def GetMDS():

    mds = manifold.MDS(n_components=2, eps=1e-6)
    distance_matrix = euclidean_distances(g_mdsInput)
    mds.fit_transform(distance_matrix)
    pos = mds.embedding_

    MDSReducedData = []
    rowDict = dict()
    for MDSIndex in range(len(pos)):
        rowDict = dict()
        for x in range(2):
            rowDict[g_featureComponentNames[x]] = pos[MDSIndex][x]
        MDSReducedData.append(rowDict)

    json_games_MDSE = json.dumps(MDSReducedData, default=json_util.default)
    return json_games_MDSE






@app.route("/games/mdsEuclidean/random")
def GetMDSRandom():


    mds = manifold.MDS(n_components=2, eps=1e-6)
    distance_matrix = euclidean_distances(g_randomMdsInput)
    mds.fit_transform(distance_matrix)
    pos = mds.embedding_


    pos = mds.embedding_

    MDSReducedData = []
    rowDict = dict()
    for MDSIndex in range(len(pos)):
        rowDict = dict()
        for x in range(2):
            rowDict[g_randomFeatureComponentNames[x]] = pos[MDSIndex][x]
        MDSReducedData.append(rowDict)

    json_games_MDSE_random = json.dumps(MDSReducedData, default=json_util.default)
    return json_games_MDSE_random






@app.route("/games/mdsCorrelation")
def GetMDSC():
    mdsIn = []
    for x in range(2):
        mdsIn.append([row[x] for row in g_odecimatedData])

    inputMatrix = np.array(mdsIn).transpose()
    distance_matrix = pairwise_distances(inputMatrix, metric='correlation')
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', eps=100,  max_iter=3000, n_jobs=1)
    mds.fit_transform(distance_matrix)

    pos = mds.embedding_

    MDSReducedData = []
    rowDict = dict()
    for MDSIndex in range(len(pos)):
        rowDict = dict()
        for x in range(2):
            rowDict[g_featureComponentNames[x]] = pos[MDSIndex][x]
        MDSReducedData.append(rowDict)

    json_games_MDSC = json.dumps(MDSReducedData, default=json_util.default)
    return json_games_MDSC


@app.route("/games/mdsCorrelation/random")
def GetMDSCRandom():
    mdsIn = []
    for x in range(2):
        mdsIn.append([row[x] for row in g_oRandomdecimatedData])

    inputMatrix = np.array(mdsIn).transpose()
    distance_matrix = pairwise_distances(inputMatrix, metric='correlation')
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', eps=10,  max_iter=3000, n_jobs=1)
    mds.fit_transform(distance_matrix)

    pos = mds.embedding_

    MDSReducedData = []
    rowDict = dict()
    for MDSIndex in range(len(pos)):
        rowDict = dict()
        for x in range(2):
            rowDict[g_randomFeatureComponentNames[x]] = pos[MDSIndex][x]
        MDSReducedData.append(rowDict)

    json_games_MDSC_random = json.dumps(MDSReducedData, default=json_util.default)
    return json_games_MDSC_random



@app.route("/games/scatterplotMatrix")
def scatterplotMatrix():
    pca = PCA(n_components=3)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    decimatedDataComponents = pipeline.fit_transform(g_pcaInputData)
    featureComponentNames = ['Component1', 'Component2', 'Component3']

    dimensionReducedData = []
    rowDict = dict()
    for decimatedIndex in range(len(decimatedDataComponents)):
        rowDict = dict()
        for x in range(3):
            rowDict[featureComponentNames[x]] = decimatedDataComponents[decimatedIndex][x]
        dimensionReducedData.append(rowDict)

    json_games_scatterplot_matrix = json.dumps(dimensionReducedData, default=json_util.default)
    return json_games_scatterplot_matrix



@app.route("/games/scatterplotMatrix/random")
def scatterplotMatrixRandom():
    pca = PCA(n_components=3)
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    decimatedDataComponents = pipeline.fit_transform(g_pcaRandomInputData)
    featureComponentNames = ['Component1', 'Component2', 'Component3']

    dimensionReducedData = []
    rowDict = dict()
    for decimatedIndex in range(len(decimatedDataComponents)):
        rowDict = dict()
        for x in range(3):
            rowDict[featureComponentNames[x]] = decimatedDataComponents[decimatedIndex][x]
        dimensionReducedData.append(rowDict)

    json_games_scatterplot_matrix_random = json.dumps(dimensionReducedData, default=json_util.default)
    return json_games_scatterplot_matrix_random


if __name__ == '__main__':
    app.run()