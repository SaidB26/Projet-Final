
# 1- Installation et chargement des Packages 

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(tidytext)) install.packages("tidytext")
if(!require(wordcloud)) install.packages("wordcloud")
if(!require(RColorBrewer)) install.packages("RColorBrewer")
if(!require(tm)) install.packages("tm")
if(!require(caret)) install.packages("caret")
if(!require(e1071)) install.packages("e1071")     
if(!require(LiblineaR)) install.packages("LiblineaR")
if(!require(ranger)) install.packages("ranger")
if(!require(pROC)) install.packages("pROC")


library(tidyverse)
library(stringr)
library(tidytext)
library(wordcloud)
library(RColorBrewer)
library(tm)
library(caret)
library(e1071)
library(LiblineaR)
library(ranger)
library(pROC)

set.seed(123) 




#2- Importation et nettoyage des données 

# Donnés Brutes
chemin_fichier <- "~/1402 final/train.ft.txt" 

# Lecture des 200000 premières lignes
texte_brut <- read_lines(chemin_fichier, n_max = 200000)

# Création du tableau et sélection de 50 000 avis au hasard
donnees_brutes <- tibble(texte_complet = texte_brut)
donnees_echantillon <- sample_n(donnees_brutes, 50000)

# Nettoyage de la mémoire (RAM) pour liberer les ressource pour la vectorisation
rm(texte_brut, donnees_brutes); gc()

# Nettoyage du texte 
donnees_propres <- donnees_echantillon %>%
  mutate(
    # 1. Créer la colonne sentiment (1 = Positif, 0 = Négatif)
    sentiment = if_else(str_detect(texte_complet, "__label__2"), 1, 0),
    
    # 2. Enlever le label du texte
    avis_texte = str_remove(texte_complet, "__label__\\d\\s"),
    
    # 3. Mettre en minuscule
    avis_propre = str_to_lower(avis_texte),
    
    
    # 4. Retrait des STOPWORDS (Mots vides)
    avis_propre = removeWords(avis_propre, stopwords("english")),
    
    # 5. Enlever la ponctuation et les chiffres 
    avis_propre = str_replace_all(avis_propre, "[[:punct:]]", " "),
    avis_propre = str_replace_all(avis_propre, "[[:digit:]]", ""),
    
    # 6. Enlever les espaces inutiles créés par les suppressions
    avis_propre = str_squish(avis_propre)
  ) %>%
  select(sentiment, avis_propre)

# 1. DISTRIBUTION DES LABELS 
graphique_distrib <- donnees_propres %>%
  mutate(label = ifelse(sentiment == 1, "Positif", "Négatif")) %>%
  ggplot(aes(x = label, fill = label)) +
  geom_bar() +
  scale_fill_manual(values = c("Positif" = "#00BFC4", "Négatif" = "#F8766D")) +
  labs(title = "Équilibre des données", x = "", y = "Nombre d'avis") +
  theme_minimal()

print(graphique_distrib)



# 2. LONGUEUR DES AVIS (Histogramme)
donnees_longueur <- donnees_propres %>%
  mutate(
    # On compte le nombre de mots 
    nb_mots = str_count(avis_propre, "\\w+"),
    label = ifelse(sentiment == 1, "Positif", "Négatif")
  )

graphique_longueur <- ggplot(donnees_longueur, aes(x = nb_mots, fill = label)) +
  geom_histogram(bins = 30, color = "white", alpha = 0.7) +
  facet_wrap(~label) + 
  labs(title = "Longueur des avis selon le sentiment", x = "Nombre de mots", y = "Fréquence") +
  theme_minimal()

print(graphique_longueur)



# 3. PREPARATION DES MOTS (Tokenisation)

data("stop_words") 

mots_par_sentiment <- donnees_propres %>%
  # On ajoute une colonne ID pour s'y retrouver
  mutate(id_avis = row_number()) %>%
  # Découpage : une ligne par mot
  unnest_tokens(mot, avis_propre) %>%
  # Nettoyage : on enlève les mots vides
  anti_join(stop_words, by = c("mot" = "word")) %>%
  # Comptage : On compte combien de fois chaque mot apparaît par sentiment
  count(sentiment, mot, sort = TRUE)

# 4. TOP MOTS (Barplots)
# les 10 mots les plus fréquents pour chaque sentiment 

top_mots <- mots_par_sentiment %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10) %>% 
  ungroup() %>%
  mutate(
    label = ifelse(sentiment == 1, "Positif", "Négatif"),
    mot = reorder_within(mot, n, sentiment) 
  )

graphique_top_mots <- ggplot(top_mots, aes(x = mot, y = n, fill = label)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~label, scales = "free_y") +
  coord_flip() + 
  scale_x_reordered() +
  labs(title = "Mots les plus fréquents", x = NULL, y = "Nombre d'apparitions") +
  theme_minimal()

print(graphique_top_mots)



# NUAGES DE MOTS 

# A. NUAGE NEGATIF
mots_negatifs <- mots_par_sentiment %>% 
  filter(sentiment == 0)

wordcloud(words = mots_negatifs$mot, freq = mots_negatifs$n, min.freq = 10,
          max.words = 80, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))

# B. NUAGE POSITIF 

mots_positifs <- mots_par_sentiment %>% 
  filter(sentiment == 1)

wordcloud(words = mots_positifs$mot, freq = mots_positifs$n, min.freq = 10,
          max.words = 80, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Set2"))

#nettoyage de la RAM
rm(donnees_longueur, top_mots, mots_negatifs, mots_positifs, mots_par_sentiment)gc()


#3- VECTORISATION 


# Création du Corpus
corpus <- VCorpus(VectorSource(donnees_propres$avis_propre))

# Création de la Matrice Document-Terme (DTM)
matrice_dtm <- DocumentTermMatrix(corpus)

# On supprime les mots rares 
matrice_dtm_legere <- removeSparseTerms(matrice_dtm, 0.99)



# Conversion en tableau pour les modèles
donnees_modele <- as.data.frame(as.matrix(matrice_dtm_legere))

# On remet la colonne sentiment en format "Facteur" (Catégorie)
donnees_modele$sentiment <- as.factor(donnees_propres$sentiment)
levels(donnees_modele$sentiment) <- c("Negatif", "Positif")

# Nettoyage mémoire
rm(corpus, matrice_dtm, donnees_echantillon, donnees_propres); gc()


# 5 : MODELISATION 


# Split Train/Test
indices <- createDataPartition(donnees_modele$sentiment, p = 0.8, list = FALSE)
jeu_train <- donnees_modele[indices, ]
jeu_test  <- donnees_modele[-indices, ]


# NAIVE BAYES

modele_nb <- naiveBayes(sentiment ~ ., data = jeu_train)

# 1. Prédiction de Classe (Pour F1-Score)
pred_class_nb <- predict(modele_nb, jeu_test)
conf_nb <- confusionMatrix(pred_class_nb, jeu_test$sentiment, mode = "prec_recall")

# 2. Prédiction de Probabilité (Pour ROC)
pred_prob_nb <- predict(modele_nb, jeu_test, type = "raw")[, "Positif"]
roc_nb <- roc(jeu_test$sentiment, pred_prob_nb, levels = c("Negatif", "Positif"))

print(conf_nb)



# SVM 

# Préparation matrices
x_train <- as.matrix(jeu_train[, -ncol(jeu_train)])
y_train <- jeu_train$sentiment
x_test  <- as.matrix(jeu_test[, -ncol(jeu_test)])

# Entraînement
modele_svm <- LiblineaR(data = x_train, target = y_train, type = 1, cost = 1)

# 1. Prédiction 
pred_svm_obj <- predict(modele_svm, newx = x_test, decisionValues = TRUE)

# Classe
pred_class_svm <- pred_svm_obj$predictions
conf_svm <- confusionMatrix(pred_class_svm, jeu_test$sentiment, mode = "prec_recall")

# Probabilité 
score_svm <- pred_svm_obj$decisionValues[,1]
roc_svm <- roc(jeu_test$sentiment, score_svm, levels = c("Negatif", "Positif"))

print(conf_svm)


# RANDOM FOREST 
# Correction des noms de colonnes
colnames(jeu_train) <- make.names(colnames(jeu_train), unique = TRUE)
colnames(jeu_test)         <- make.names(colnames(jeu_test), unique = TRUE)

#entrainement

modele_rf <- ranger(sentiment ~ ., data = jeu_train, num.trees = 100, probability = TRUE)



# Prédictions 
pred_rf_obj <- predict(modele_rf, data = jeu_test)$predictions
pred_prob_rf <- pred_rf_obj[, "Positif"]


pred_class_rf <- as.factor(ifelse(pred_prob_rf > 0.5, "Positif", "Negatif"))
conf_rf <- confusionMatrix(pred_class_rf, jeu_test$sentiment, mode = "prec_recall")

# 2. Courbe ROC
roc_rf <- roc(jeu_test$sentiment, pred_prob_rf, levels = c("Negatif", "Positif"))

print(conf_rf)


# 6 : Evaluation et Comparaison

# 1. Fonction pour extraire F1 et AUC
get_metrics <- function(conf, roc_obj, nom) {
  data.frame(
    Modele = nom,
    Accuracy = round(conf$overall["Accuracy"], 4),
    F1_Score = round(conf$byClass["F1"], 4),    
    AUC      = round(auc(roc_obj), 4)         
  )
}

# 2. Création du tableau
resultats <- rbind(
  get_metrics(conf_nb, roc_nb, "Naive Bayes"),
  get_metrics(conf_svm, roc_svm, "SVM"),
  get_metrics(conf_rf, roc_rf, "Random Forest")
)

print(resultats)



# 3. DESSIN DES COURBES ROC 


g_roc <- ggroc(list("Naive Bayes" = roc_nb, "SVM" = roc_svm, "Random Forest" = roc_rf)) +
  geom_abline(intercept = 1, slope = 1, color = "gray", linetype = "dashed") + 
  labs(title = "Comparaison des Courbes ROC", 
       x = "Spécificité (Taux de vrais négatifs)", 
       y = "Sensibilité (Taux de vrais positifs)") +
  theme_minimal() +
  scale_color_manual(values = c("Naive Bayes"="#F8766D", "SVM"="#00BA38", "Random Forest"="#619CFF"))

print(g_roc)

