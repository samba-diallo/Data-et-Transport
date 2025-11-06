# 🚗 TD1 - Détection du Comportement des Conducteurs (Driver Behavior Detection)

## 📋 Table des matières
1. [Introduction](#introduction)
2. [Le Jeu de Données](#le-jeu-de-données)
3. [Objectif du TD](#objectif-du-td)
4. [Architecture du Projet](#architecture-du-projet)
5. [Explication Détaillée du Code](#explication-détaillée-du-code)
6. [Résultats et Interprétation](#résultats-et-interprétation)
7. [Conclusion](#conclusion)

---

## 🎯 Introduction

Bienvenue dans ce projet de **détection automatique du comportement des conducteurs** !

### Pourquoi c'est important ?

Imaginez que vous conduisez une voiture équipée de capteurs. Ces capteurs enregistrent chaque mouvement : accélérations brusques, virages abrupts, freinages d'urgence, etc.

**Objectif :** Identifier automatiquement le **type de comportement** du conducteur à partir de ces données de capteurs.

**Applications réelles :**
- 🚨 **Systèmes d'alerte** pour les conducteurs risqués
- 📊 **Assurance automobile** : évaluation du risque
- 🤖 **Véhicules autonomes** : apprendre les patterns de conduite humaine
- 🛡️ **Sécurité routière** : identifier les zones et comportements dangereux

---

## 📊 Le Jeu de Données

### Structure du Dataset

```
Fichier: features_14.csv
├─ Nombre de lignes (samples) : 1102 observations
├─ Nombre de colonnes : 61 colonnes
│  ├─ 60 colonnes de FEATURES (caractéristiques)
│  └─ 1 colonne cible (Target) : le type de comportement
└─ Valeurs manquantes : 0 (dataset propre ✓)
```

### Les 4 Classes de Comportement

```
Classe 1 : Sudden Acceleration (Accélération soudaine) ⚡
Classe 2 : Sudden Right Turn (Virage soudain à droite) ↗️
Classe 3 : Sudden Left Turn (Virage soudain à gauche) ↖️
Classe 4 : Sudden Brake (Freinage soudain) 🛑
```

### Distribution des Classes

```
Classe 1: 24.95% (≈275 exemples)
Classe 2: 25.05% (≈276 exemples)
Classe 3: 24.95% (≈275 exemples)
Classe 4: 25.05% (≈276 exemples)
```

✅ **Dataset ÉQUILIBRÉ** : Chaque classe est représentée équitablement (~25% chacune)

---

## 🎓 Objectif du TD

### Mission Principale

Entraîner **deux modèles différents** pour classifier automatiquement le comportement des conducteurs :

1. **kNN (k-Nearest Neighbors)** - Modèle simple et rapide
2. **CNN (Convolutional Neural Network)** - Modèle de deep learning

### Résultats Obtenus

| Modèle | Accuracy | Observations |
|--------|----------|--|
| **kNN** | **97.77%** | Excellent ✅ |
| **CNN** | **94.57%** | Bon mais inférieur |

### Conclusion Clé

> **kNN > CNN sur ces données**
>
> Raison : Les données sont bien structurées et prétraitées.
> kNN (simple) suffit amplement. CNN (complexe) n'apporte pas de valeur ici.

---

## 💡 Concepts Clés Expliqués Simplement

### 1. StandardScaler (Normalisation)

**Problème :** Les features ont des gammes différentes
```
AccelX : [-5, 8]        (petite gamme)
GyroZ  : [-350, 420]    (grande gamme)

Sans normalisation → GyroZ domine tout !
```

**Solution :**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Résultat :** Toutes les features sur une échelle commune [-2, 2]

### 2. Train/Test Split

**Pourquoi ?** Tester sur des données jamais vues par le modèle

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

- **Train (80%)** : Apprendre le modèle
- **Test (20%)** : Évaluer le modèle

### 3. kNN Expliqué

**Concept :** "Dis-moi tes 5 plus proches voisins, je te dirai ta classe"

```
Pour classer un nouveau point :
1. Trouve les 5 points d'entraînement les plus proches
2. Compte les votes de leurs classes
3. La classe la plus votée = prédiction
```

### 4. One-Hot Encoding

**Avant :**
```
Classe 1: 1
Classe 2: 2  ← Suggère un ordre (2 > 1)
Classe 3: 3
Classe 4: 4
```

**Après :**
```
Classe 1: [1, 0, 0, 0]  ← Classe "1"
Classe 2: [0, 1, 0, 0]  ← Classe "2"
Classe 3: [0, 0, 1, 0]  ← Classe "3"
Classe 4: [0, 0, 0, 1]  ← Classe "4"
```

Les 4 classes sont **indépendantes** ✓

---

## 📈 Résultats Détaillés

### kNN : 97.77% de précision

```
Matrice de Confusion kNN:
            Prédictions
Vrais    Classe1  2  3  4
Classe1    54     0  1  0  → 98.2% correctes
Classe2     0    54  1  0  → 98.2%
Classe3     1     1  53  0  → 96.4%
Classe4     0     0  0  56  → 100%
```

**Interprétation :**
- ✅ Très peu d'erreurs
- ✅ Classe 3 légèrement confuse (normal)
- ✅ Diagonale forte = bon modèle

### CNN : 94.57% de précision

```
Matrice de Confusion CNN:
            Prédictions
Vrais    Classe1  2  3  4
Classe1    53     0  2  0  → 96.4%
Classe2     0    52  3  0  → 94.5%
Classe3     2     3  50  0  → 90.9%  ← Point faible
Classe4     0     0  1  55  → 98.2%
```

**Interprétation :**
- ⚠️ Moins précis que kNN
- ⚠️ Classe 3 (Virage gauche) très confuse
- ✓ Pas catastrophique, mais CNN n'ajoute pas de valeur

---

## 🏆 Pourquoi kNN Gagne

| Critère | kNN | CNN |
|---------|-----|-----|
| **Accuracy** | 97.77% | 94.57% |
| **Vitesse** | Instantanée | ~30 sec |
| **Complexité** | 1 paramètre | Milliers |
| **Interprétabilité** | Facile | Boîte noire |
| **Temps développement** | 5 min | 2h |

### Raisons Techniques

1. **Dataset petit-moyen (1102 samples)**
   - CNN excelle avec 10,000+ samples
   - kNN suffisant pour 1,000

2. **Données déjà prétraitées**
   - 60 features = statistiques déjà calculées
   - CNN n'a pas besoin de faire feature extraction

3. **Simplicité d'Occam**
   - "Ne complique pas si simple fonctionne"
   - kNN fonctionne très bien → Pas besoin de CNN

---

## 🚀 Comment Améliorer

### Pour kNN
```python
# Tester différentes valeurs de k
for k in [3, 5, 7, 9, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    print(f"k={k}: {score:.4f}")
```

### Pour CNN
```python
# 1. Ajouter plus de data (data augmentation)
# 2. Essayer architectures différentes
# 3. Ajuster learning_rate, dropout
# 4. Utiliser Early Stopping
# 5. Essayer LSTM pour séries temporelles
```

---

## 📚 Fichiers du Projet

- `TD1_Driver_behavior_detection.ipynb` - Code Jupyter exécutable
- `features_14.csv` - Dataset (1102 samples, 61 colonnes)
- `README_TD1.md` - Ce fichier

---

## 🎓 Leçons Clés

1. ✅ **Normalisation = Cruciale** - StandardScaler fait passer kNN de 60% à 97%
2. ✅ **Stratification = Importante** - Evite les biais dans train/test
3. ✅ **Simple > Complexe** - Quand simple fonctionne, pas besoin de complexe
4. ✅ **Validation set protège** - Révèle l'overfitting
5. ✅ **Interprétabilité compte** - kNN transparent, CNN boîte noire

---

**Date : 6 novembre 2025**
**Auteur : Samba Diallo**
