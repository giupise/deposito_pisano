import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1 prendo i dati dal file csv
df = pd.read_csv("Mall_Customers.csv")

# nel dataframe -> Annual Income e Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 2 standardizzo 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3 calcolo silhouette score per k da 2 a 10
# con k = 1 faccio confluire tutto nello stesso cluster, per cui parto da 2
# mi fermo a 10 per non far diventare i cluster troppo piccoli
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# grafico silhouette score
plt.figure(figsize=(8, 5))
sns.lineplot(x=list(K), y=silhouette_scores, marker='x')
plt.title("Silhouette Score al variare di k")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# 4 k migliore (con massimizzazione del silhouette score)
best_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"Miglior numero di cluster secondo silhouette: {best_k}")

# dal grafico:
# k=2 -> silhouette basso -> cluster non ben separati
# da k=3 a k=5 lo score aumenta -> struttura cluster miglior3.
# picco massimo è k=5
# da k=6-> il punteggio scende rispetto a k=5 -> i cluster aggiuntivi non migliorano la segmentazione
# la funzione di massimizzazione coincide con l'analisi del grafico

# 5allenamento KMeans con k max = 5
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# aggiungo i cluster al dataframe
df['Cluster'] = labels

# 6. Scatter plot 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X['Annual Income (k$)'], 
    y=X['Spending Score (1-100)'], 
    hue=df['Cluster'], 
    palette="Set2", 
    s=80
)

# centroidi
centroids = scaler.inverse_transform(kmeans.cluster_centers_) # per avere le unità originali
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c="black", s=200, marker="X", label="Centroidi"
)

plt.title(f"K-Means Clustering con k={best_k}")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


# 7 interpretazione cluster

# cluster 0 (verde acqua, al centro)
# reddito medio
# spending score medio
# reddito e spesa nella media

# cluster 1 (arancione, in alto a destra)
# alto reddito
# alto spending score
# sono i clienti alto spendenti

# cluster 2 (blu, in alto a sinistra)
# basso reddito
# alto spending score
# clienti che pur non avendo grandi entrate spendono molto

# cluster 3 (rosa, in basso a destra)
# alto reddito
# basso spending score
# clienti con potenziale economico alto, ma che spendono poco

# cluster 4 (giallo/verde, in basso a sinistra)
# basso reddito
# basso spending score
# clienti a basso valore, con capacità di spesa limitata e basso interesse
