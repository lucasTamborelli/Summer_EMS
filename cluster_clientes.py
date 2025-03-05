import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# Leitura
df = pd.read_csv('2024_cluster.csv')

# Dados Separados e Removidos
quant_estado = df.groupby('estado').count().iloc[:, 0]
df_cd = df[['cd_cliente']].copy()
df_sx = df[['sexo']].copy()
df_estado = df[['estado']].copy()
df_quants = df[['quant_30', 'quant_60', 'quant_8']].copy()
colunas_remov = ['estado', 'cd_cliente', 'sexo', 'quant_30', 'quant_60', 'quant_8']
df = df.drop(colunas_remov, axis=1)

# Boxplot Antes da Normalizacao
plt.figure()
ax = sns.boxplot(data = df)
plt.show()

# Normalizacao
obj_normalizacao = MinMaxScaler().fit(df)
df_norm = obj_normalizacao.transform(df)

# So os dados que serao utilizados para a clusterizacao
colunas = ['nr_compras','ticket_medio','desconto'] 
df_norm = pd.DataFrame(df_norm, columns = colunas)
print(df_norm.describe())

# Boxplot Depois da Normalizacao
plt.figure()
ax = sns.boxplot(data = df_norm)
plt.show()

# Obs:
# Fração dos dados para treinamento e teste nao foi utilizada (so 24k linhas)
# Ex: amostra1, amostra2 = train_test_split(df, train_size = 0.1)


# Gráfico Elbow (com porcentagem da variancia)
# Argumento(!?) para a quantidade de K no KMeans

k_range = range(1, 8)
k_means_var = [KMeans(n_clusters = k).fit(df_norm) for k in k_range]

centroids = [X.cluster_centers_ for X in k_means_var]
k_euclidiana = [cdist(df_norm, cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis = 1) for ke in k_euclidiana]

soma_q_intra_cluster = [sum(d**2) for d in dist]
soma_tot = sum(pdist(df_norm)**2)/df_norm.shape[0]
soma_q_inter_cluster = soma_tot - soma_q_intra_cluster 

fig = plt.figure()
ax =fig.add_subplot(111)
ax.plot(k_range, soma_q_inter_cluster/soma_tot * 100, '-o')
ax.set_ylim(0,100)
plt.show()


# Discussao: K = 5 (?) 
###
sc_ss = []
sc_db = []
vet = []
for j in range (2,7):
    vet.append(j)
    modelo_1 = KMeans(n_clusters = j)
    modelo_1.fit(df_norm)
    labels = modelo_1.labels_

    sc_ss.append(silhouette_score(df_norm, labels, metric = 'euclidean'))
    sc_db.append(davies_bouldin_score(df_norm, labels))
    
# Davies-Bouldin
plt.plot(vet, sc_db, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Índice Bouldin')
plt.show()

# Silhuette Score
plt.plot(vet, sc_ss, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Score de Silhueta')
plt.show()
###

# Modelo com K =5 clusters
modelo_1 = KMeans(n_clusters = 5)
modelo_1.fit(df_norm)
labels = modelo_1.labels_

print(df_norm.head())
print(df_norm.shape)


# Desnormalizando
nomes = ['nr_compras','ticket_medio', 'desconto']
df_final = obj_normalizacao.inverse_transform(df_norm)
df_final = pd.DataFrame(df_final, columns = nomes)

# print(df_final.head(10))

df_final['cluster'] = modelo_1.labels_
df_final['cd_cliente'] = df_cd
df_final['estado'] = df_estado
df_final['quant_30'] = df_quants['quant_30']
df_final['quant_8'] = df_quants['quant_8']
df_final['quant_60'] = df_quants['quant_60']


# Visualizacao dos dados
print(df_final.head(50))
print(df_final.groupby('cluster')['nr_compras'].mean())
print(df_final.groupby('cluster')['ticket_medio'].mean())
print(df_final.groupby('cluster')['desconto'].mean())
print(df_final.groupby('cluster')['nr_compras'].count())
print(df_final.groupby('cluster')['quant_30'].sum())
print(df_final.groupby('cluster')['quant_8'].sum())
print(df_final.groupby('cluster')['quant_60'].sum())

# print('Quantidade dos cluster por estado: tentativa 1\n', df_final.groupby(['estado', 'cluster'])['cluster'].count())

print('Quantidade de clusters por estado: tentativa 2\n', df_final.groupby('estado')['cluster'].value_counts().unstack().fillna(0))
print('Quantidade de pessoas por estado', print(quant_estado))


# Visualizacao grafica da clusterizacao

#Reduzindo p/2 dimensoes
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_norm)

# Figura 1
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.colorbar(label='Clusterização')
plt.show()


# Figura 2
modelo_1.fit(df_pca)

x_min, x_max = df_pca[:, 0].min() - 5, df_pca[:, 0].max() -1
y_min, y_max = df_pca[:, 1].min() + 1, df_pca[:, 1].max() +5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = modelo_1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize= (8,6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], color='black', s=50, alpha=0.5)
centroids = modelo_1.cluster_centers_
inert = modelo_1.inertia_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidths =3, color = 'r', zorder = 8)
plt.show()
