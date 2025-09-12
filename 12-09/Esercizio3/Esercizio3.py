import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# 1 carico i dati
df = pd.read_csv("train.csv")
X = df.drop(columns=["label"])
y = df["label"]

# 2 split train/test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3 scalo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4 calcolo PCA solo sul train
pca = PCA(n_components=0.95, whiten=True)   # numero di componenti che spiegano il 95% della varianza
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("numero componenti PCA:", pca.n_components_)

# 5 alleno decision tree con PCA
tree_pca = DecisionTreeClassifier(random_state=42)
tree_pca.fit(X_train_pca, y_train)

# 6 valuto su test, stampando accuracy e matrice di confusione
y_pred_pca = tree_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
cm_pca = confusion_matrix(y_test, y_pred_pca)

print("accuratezza con PCA:", acc_pca)
print("matrice di confusione con PCA:\n", cm_pca)

# tempo di training albero con PCA
start = time.perf_counter()
tree_pca = DecisionTreeClassifier(random_state=42)
tree_pca.fit(X_train_pca, y_train)
t_train_pca = time.perf_counter() - start
print(f"training time (con PCA): {t_train_pca:.4f} s")

# 7 confronto con modello senza PCA
tree_plain = DecisionTreeClassifier(random_state=42)
tree_plain.fit(X_train_scaled, y_train)
y_pred_plain = tree_plain.predict(X_test_scaled)
acc_plain = accuracy_score(y_test, y_pred_plain)
cm_plain = confusion_matrix(y_test, y_pred_plain)

print("\naccuratezza senza PCA:", acc_plain)
print("matrice di confusione senza PCA:\n", cm_plain)

start = time.perf_counter()
tree_plain = DecisionTreeClassifier(random_state=42)
tree_plain.fit(X_train_scaled, y_train)
t_train_plain = time.perf_counter() - start
print(f"training time (senza PCA): {t_train_plain:.4f} s")

# 8 Interpretazione

# con PCA
# numero componenti PCA: 318 su 784 bastano a spiegare il 95% della varianza
# accuratezza con PCA: 0.8116666666666666 -> buona
# tempo di training maggiore  

# senza PCA
# accuratezza senza PCA: 0.8552380952380952 -> più alta rispetto alla precedente -> può essere che riducendo le dimensioni ho perso qualità con PCA?
# tempo di training minore 