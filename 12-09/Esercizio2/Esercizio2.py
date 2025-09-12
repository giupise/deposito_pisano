import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# 2 carico dati
df = pd.read_csv("Wholesale customers data.csv")

# 3 tengo solo le feature numeriche utili
X = df.drop(columns=['Channel', 'Region'])

# 4 standardizzo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5 parametri
eps_value = 1.5
min_samples_list = [3, 5, 8]


# creo una maschera che esclude gli outlier (-1)  
# se sono rimasti solo outlier silhouette non calcolabile  
# tengo solo i dati e le etichette dei non-outlier  
# conto quanti elementi ha ciascun cluster  
# tengo solo i cluster con almeno 2 punti  
# se resta meno di 2 cluster validi silhouette non calcolabile  
# calcolo silhouette score solo sui cluster validi (no outlier, no cluster singoli)  

def safe_silhouette(X_scaled, labels):
    mask = labels != -1
    if not np.any(mask):
        return None
    Xc = X_scaled[mask]
    yc = labels[mask]
    # rimuovo cluster con 1 solo punto
    uniq, cnt = np.unique(yc, return_counts=True)
    keep = uniq[cnt >= 2]
    if len(keep) < 2:
        return None
    kmask = np.isin(yc, keep)
    return silhouette_score(Xc[kmask], yc[kmask])

# 7 prova dei tre min_samples
summary = []
models = {}
# applico DBSCAN con eps fisso e il valore corrente di min_samples  
# etichette dei cluster trovati (-1 indica outlier)  
# numero di cluster validi tolto l’eventuale -1  
# numero di outlier (punti etichettati -1)  
# silhouette score calcolato in modo sicuro (escludendo outlier e cluster singoli)  
# salvo i risultati in una lista di dizionari  
# salvo anche il modello DBSCAN e le etichette per questo min_samples  
for m in min_samples_list:
    db = DBSCAN(eps=eps_value, min_samples=m).fit(X_scaled)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_out = int(np.sum(labels == -1))
    sil = safe_silhouette(X_scaled, labels)
    summary.append({"min_samples": m, "n_clusters": n_clusters, "n_outliers": n_out,
                    "silhouette": None if sil is None else round(sil, 3)})
    models[m] = (db, labels)

summary_df = pd.DataFrame(summary).sort_values(["silhouette"], ascending=False, na_position="last")
print("confronto configurazioni (eps fisso):")
print(summary_df.to_string(index=False))

# 8 scelta semplice: massima silhouette disponibile, altrimenti più cluster e meno outlier
# se esiste almeno una configurazione con silhouette calcolata,
# scelgo quella con il valore massimo di silhouette
# altrimenti (se silhouette non definita per nessuna configurazione),
# scelgo la configurazione con il maggior numero di cluster
# e a parità di cluster con il minor numero di outlier
if summary_df["silhouette"].notna().any():
    best_m = summary_df.dropna(subset=["silhouette"]).iloc[0]["min_samples"]
else:
    best_m = summary_df.sort_values(["n_clusters", "n_outliers"], ascending=[False, True]).iloc[0]["min_samples"]

print(f"\nconfigurazione scelta -> eps={eps_value}, min_samples={int(best_m)}")

# 9 applico la migliore e preparo l’interpretazione
# recupero il modello e le etichette corrispondenti al min_samples scelto
# aggiungo la colonna "Cluster" al dataframe
# considero solo i punti non outlier (Cluster != -1)
# se non ci sono cluster validi o c’è solo un cluster, stampo un messaggio
# altrimenti calcolo le medie delle variabili per cluster
# e stampo la tabella riassuntiva (utile per l’interpretazione)
best_db, best_labels = models[int(best_m)]
df["Cluster"] = best_labels

valid = df[df["Cluster"] != -1]
if valid.empty or valid["Cluster"].nunique() < 1:
    print("\npochi cluster utili: molti outlier o un solo cluster")
else:
    means = valid.groupby("Cluster")[X.columns].mean().round(2)
    print("\nmedie per cluster (esclusi outlier):")
    print(means.to_string())