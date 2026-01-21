# Projet-Final
Projet d'Analyse de Sentiments - Amazon Reviews
Ce répertoire contient mon projet final d'analyse de données textuelles (NLP). L'objectif est de prédire si un avis client Amazon est Positif ou Négatif en utilisant R studio.

Objectif
Analyser un échantillon de 50 000 avis pour comprendre ce qui distingue un commentaire satisfait d'un commentaire mécontent, et entraîner un modèle capable de classer automatiquement les nouveaux avis.

Source des données:
https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

Bibliothèques utilisées
Le projet utilise les packages R suivants pour le traitement de données, la visualisation et la modélisation :

tidyverse, stringr
tidytext, tm (Text Mining)
wordcloud, RColorBrewer, ggplot2 (via tidyverse), pROC
Modèles (Machine Learning) :
e1071 (Naive Bayes)
LiblineaR (SVM - Support Vector Machine)
ranger (Random Forest)
caret (Outils d'évaluation)
Méthodologie
Voici les étapes suivies dans le script (1402 script final) :

1. Importation et Nettoyage
Les données brutes étant très volumineuses, j'ai travaillé sur un échantillon aléatoire de 50 000 lignes.

Labels : Conversion des labels en binaire (1 = Positif, 0 = Négatif).
Nettoyage textuel :
Mise en minuscule.
Suppression des Stopwords (mots vides).
Suppression de la ponctuation et des chiffres.
Suppression des espaces inutiles.
2. Analyse Exploratoire (EDA)
Avant de lancer les modèles, j'ai visualisé les données pour vérifier leur qualité :

Distribution : Les données sont équilibrées (autant de positifs que de négatifs). Distribution
Longueur : Histogramme pour voir si les avis positifs sont plus longs ou plus courts. Longueur des avis
Mots fréquents : Barplots des top 10 mots pour chaque sentiment. Mots fréquents
Nuages de mots : Visualisation des termes les plus utilisés.
Négatifs: Nuages de mots négatifs
Positifs: Nuages de mots négatifs
3. Vectorisation
J'ai transformé le texte nettoyé en une matrice (Document-Term Matrix) en ne gardant que les mots qui apparaissent dans au moins 1% des avis pour alléger le modèle.

4. Modélisation et Résultats
J'ai divisé les données en Entraînement (80%) et Test (20%) pour comparer trois algorithmes.

Voici les performances obtenues (sur le jeu de test) :

Modèle	Accuracy (Précision globale)	F1-Score	AUC (Courbe ROC)
Naive Bayes	0.7716	0.7511	0.8380
SVM	0.8329	0.8269	0.9100
Random Forest	0.8120	0.8062	0.8907
5. Comparaison Graphique (ROC)
Nous avons utilisé les courbes ROC pour visualiser la performance. Plus la courbe se rapproche du coin haut-gauche, meilleur est le modèle.

Courbes ROC

Conclusion
D'après les métriques (Accuracy et AUC), le modèle SVM semble être le plus performant pour cette tâche de classification.
