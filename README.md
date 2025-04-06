# Projet CobotOps : Prédiction des Arrêts de Protection d'un Cobot UR3

## Auteur
- [Youssra KESSOU]
## Objectif
Développer un modèle de machine learning pour prédire les arrêts de protection (`Robot_protectivestop`) d'un cobot UR3 à partir des données de capteurs (courants, températures, vitesses) sur une séquence temporelle de 10 pas.

## Méthodologie

### 1. Prétraitement des Données
- **Nettoyage** :
  - Suppression des colonnes vides (`Unnamed: 24` à `34`).
  - Imputation des valeurs manquantes par la **médiane** pour les capteurs.
  - Conversion des booléens (`grip_lost`) en entiers (0/1).
- **Feature Engineering** :
  - Normalisation avec `StandardScaler`.
  - Création de séquences temporelles (fenêtres de 10 pas).

### 2. Modélisation
- **Modèles Testés** :
  - **LSTM** (TensorFlow/Keras) pour capturer les dépendances temporelles.
  - **Random Forest** (scikit-learn) comme baseline.
  - **XGBoost** avec fenêtres glissantes.
- **Optimisation** :
  - GridSearch pour Random Forest.
  - Keras Tuner pour le LSTM.
- **Métriques** :
  - Recall, F1-score, AUC-ROC (priorité sur la détection des arrêts).

### 3. Déploiement avec Docker
- **Construire l'image** :
  docker build -t cobot-api .
- **Lancer le conteneur** :
  docker run -p 5000:5000 --name cobot-container cobot-api
- **Arrêter le conteneur** :
  docker stop cobot-container

### 4. Résultats Techniques
- **Performances des Modèles** :

Modèle      Recall (Classe 1)     F1-Score (Classe 1)	
XGBoost	    78%	                  0.72	
LSTM	      82%	                  0.75	