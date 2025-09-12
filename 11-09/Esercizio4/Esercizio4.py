from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

# carico il dataset
df = pd.read_csv("creditcard.csv") 

# divido input e target
X = df.drop("Class", axis=1) 
y = df["Class"]

# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
auc_tree = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")

print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")

# Plot dei risultati cross-validation
plt.figure(figsize=(6,4))
plt.boxplot(auc_tree, vert=False, patch_artist=True)
plt.title("Decision Tree ROC-AUC (5-fold CV)")
plt.xlabel("ROC-AUC")
plt.yticks([1], ["Decision Tree"])
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()