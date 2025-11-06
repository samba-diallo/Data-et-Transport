# 📚 Data et Transport - Travaux Dirigés

[![GitHub](https://img.shields.io/badge/GitHub-Data--et--Transport-blue?logo=github)](https://github.com/samba-diallo/Data-et-Transport)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Bienvenue dans ce repository dédié aux **Travaux Dirigés (TD)** du cours **Data et Transport** !

## 🎯 Objectif Global

Ce projet couvre les concepts fondamentaux du **Machine Learning et de l'Analyse de Données** appliqués au domaine du **Transport et de la Sécurité Routière**.

## 📖 Structure du Repository

```
Data-et-Transport/
├── TD1_Driver_Behavior_Detection/
│   ├── README_TD1.md                          ← Explications complètes
│   ├── TD1_Driver_behavior_detection.ipynb    ← Code exécutable
│   └── features_14.csv                        ← Dataset
│
├── TD2_Traffic_Prediction/
│   ├── README_TD2.md
│   └── TD2_Traffic_prediction.ipynb
│
├── TD3_Accident_Detection/
│   ├── README_TD3.md
│   ├── TD3_Accident_detection.ipynb
│   └── data/
│       ├── caract-2023.csv
│       ├── lieux-2023.csv
│       ├── usagers-2023.csv
│       └── vehicules-2023.csv
│
├── TD4_Object_Detection/
│   ├── README_TD4.md
│   └── TD4_Object_detection.ipynb
│
└── README.md                                  ← Ce fichier
```

## 🚗 TD1 - Détection du Comportement des Conducteurs

### 📌 Description
Classification de 4 types de comportements de conduite à partir de données de capteurs.

### 📊 Dataset
- **Samples** : 1102
- **Features** : 60 (statistiques de capteurs)
- **Classes** : 4 (Accélération soudaine, Virage droite, Virage gauche, Freinage)
- **Distribution** : Équilibrée (≈25% chaque classe)

### 🤖 Modèles
| Modèle | Accuracy | Note |
|--------|----------|------|
| **kNN** | **97.77%** | ⭐ Gagnant |
| **CNN** | 94.57% | Bon mais inférieur |

### 📚 Documentation
Consultez [`TD1_Driver_Behavior_Detection/README_TD1.md`](./TD1_Driver_Behavior_Detection/README_TD1.md) pour :
- Explications détaillées du code
- Visualisations des résultats
- Interprétation des matrices de confusion
- Concepts clés expliqués simplement

### 🚀 Lancer le projet
```bash
jupyter notebook TD1_Driver_Behavior_Detection/TD1_Driver_behavior_detection.ipynb
```

---

## 🔧 Installation

### Prérequis
- Python 3.10+
- pip ou conda

### Dépendances
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### Cloner le repository
```bash
git clone https://github.com/samba-diallo/Data-et-Transport.git
cd Data-et-Transport
```

---

## 📚 Technologies

- **Python 3.10+** - Langage principal
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Scikit-learn** - Machine Learning
- **TensorFlow/Keras** - Deep Learning
- **Matplotlib/Seaborn** - Visualisations
- **Jupyter** - Notebooks interactifs

---

## 📚 Ressources

- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Dataset Original](https://data.mendeley.com/datasets/jj3tw8kj6h/2)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## 👨‍💼 Auteur

**Samba Diallo**
- 📧 Email: samba.diallo@edu.esiee.fr
- 🔗 GitHub: [@samba-diallo](https://github.com/samba-diallo)

---

## 📝 Licence

Ce projet est open source sous licence MIT.

---

**Dernière mise à jour : 6 novembre 2025**
