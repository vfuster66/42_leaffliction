
# 🍃 Leaffliction
> Classification automatique des maladies des feuilles de pommiers et de vignes.

---

## 📚 Présentation du projet

Le projet **Leaffliction** est une application de classification d'images, réalisée en Python 3.10.  
Son objectif est d'**identifier automatiquement** la maladie affectant une feuille de **pommier** ou de **vigne**, à partir d'une simple photo.

Ce projet repose sur un pipeline **Data Science complet**, qui comprend :  
1. L'**analyse du dataset** d'images fournies.  
2. La **génération de nouvelles images** (augmentation) pour enrichir l'apprentissage.  
3. La **transformation d'images** pour mieux comprendre les caractéristiques visuelles des maladies.  
4. L'**entraînement d'un modèle de Deep Learning** (CNN) pour la classification automatique.  
5. La **prédiction** sur de nouvelles images, avec l'affichage de résultats et de matrices de confusion.

---

## ✅ Objectifs pédagogiques
Ce projet vise à :
- 🏷️ Comprendre le **prétraitement** d'un dataset d'images et l'enrichissement par **Data Augmentation**.
- 🧠 Apprendre à **entraîner un modèle CNN** (réseau de neurones convolutif) pour la classification.
- 📈 Savoir **évaluer et interpréter les performances** d'un modèle de machine learning.
- ⚙️ Mettre en place un pipeline **automatisé**, **dockerisé**, respectant la **norme flake8**.
- ✅ Maîtriser la **validation des résultats** et l'**exportation** du modèle pour réutilisation.

---

## 🗂️ Arborescence du projet

```
42_leaffliction/
├── app/                
│   ├── Distribution.py        
│   ├── Augmentation.py        
│   ├── Transformation.py      
│   ├── train.py               
│   └── predict.py             
├── images/                    
├── output/                    
│   ├── models/                
│   ├── plots/                 
│   ├── predictions/           
│   ├── balanced/              
│   ├── augmented/             
│   └── transformations/       
├── tests/                     
├── Dockerfile                 
├── Makefile                   
├── requirements.txt           
└── README.md                  
```

---

## 🛠️ Technologies et Dépendances

Le projet **Leaffliction** repose sur des technologies open-source robustes, adaptées à la **classification d'images** et au **traitement d'images en machine learning**.

### 🐳 Docker
- **Pourquoi ?** ➔ Assurer un environnement isolé, reproductible et indépendant du système de l'utilisateur.
- Le Dockerfile configure un conteneur Python 3.10 avec TensorFlow, OpenCV, etc.

### 🐍 Python 3.10
- Langage principal du projet.
- Simplicité, lisibilité et puissance pour le traitement d'images et le deep learning.

---

## 📦 Librairies Python utilisées (requirements.txt)

| 📚 Librairie        | 🎯 Usage |
|---------------------|--------------------------------|
| **numpy**           | Manipulation efficace de tableaux de données numériques (traitement d’images). |
| **pandas** (optionnel) | Gestion structurée des données tabulaires, si besoin pour des analyses complémentaires. |
| **matplotlib**      | Visualisation : courbes d'apprentissage, histogrammes, matrices de confusion. |
| **seaborn** (optionnel) | Visualisation avancée, matrices de confusion plus lisibles si activé. |
| **opencv-python**   | Traitement d’images : lecture, transformations (crop, blur, edge detection, etc.). |
| **tensorflow / keras** | Création et entraînement du modèle CNN, prédictions sur nouvelles images. |
| **scikit-learn**    | Outils d'évaluation : matrice de confusion, calcul de précision, etc. |
| **albumentations**  | Data Augmentation avancée (non activée mais prête à l’emploi si besoin). |
| **colorama**        | Affichage coloré dans le terminal pour une meilleure lisibilité des logs. |
| **pytest**          | Exécution des tests unitaires. |
| **flake8**          | Analyse statique du code pour vérifier la conformité à la norme PEP8. |

---

## 🏁 Pipeline complet du projet

### 1️⃣ Analyse du dataset : **Comprendre les données**
```
make run-distribution
```
- Vérifier la **répartition** des images par classes.
- Générer : diagramme circulaire + histogramme.

### 2️⃣ Data Augmentation : **Enrichir le dataset**

La Data Augmentation permet de générer des variations artificielles à partir des images originales. Cela augmente la robustesse du modèle, évite l'overfitting et améliore la généralisation.

### ➤ 1. Flip (symétrie horizontale)
- **Description** : retourne l’image horizontalement, comme dans un miroir.
- **Pourquoi ?** : améliore la capacité du modèle à reconnaître des feuilles peu importe leur orientation.

### ➤ 2. Rotate (rotation de l'image)
- **Description** : effectue une rotation à 90°, 180°, ou 270°.
- **Pourquoi ?** : aide à l’identification des maladies sur des feuilles prises sous différents angles.

### ➤ 3. Skew (distorsion de perspective)
- **Description** : transforme l’image en biaisant la perspective via une matrice affine.
- **Pourquoi ?** : simule des photos prises avec des angles inclinés ou de travers.

### ➤ 4. Shear (cisaillement)
- **Description** : applique une transformation de cisaillement sur l’axe horizontal ou vertical.
- **Pourquoi ?** : renforce la tolérance du modèle à des déformations naturelles ou accidentelles des feuilles sur la photo.

### ➤ 5. Crop (recadrage central)
- **Description** : découpe une portion centrale de l’image, puis la redimensionne à la taille d’origine.
- **Pourquoi ?** : permet de se focaliser sur la zone centrale, là où les symptômes sont souvent plus apparents.

### ➤ 6. Distortion (flou gaussien)
- **Description** : applique un flou gaussien pour réduire les détails fins et lisser les variations de l’image.
- **Pourquoi ?** : améliore la robustesse du modèle sur des images floues ou de qualité réduite.

#### ➤ Commande :
```
make run-augmentation
make balance-augmented
```
- Génération de 6 augmentations par image.
- Objectif : 500 images/classe dans `output/balanced/`.

### 3️⃣ Transformations d'images (optionnel) : **Analyse visuelle**

Ces transformations sont principalement destinées à la **visualisation analytique** et à la **compréhension** des caractéristiques visuelles des feuilles.

### ➤ 1. Gaussian Blur (flou gaussien)
- **Description** : filtre l’image avec un flou gaussien (noyau de 9x9).
- **But** : réduit le bruit de l’image et accentue les grandes structures.

### ➤ 2. Edge Detection (détection des contours)
- **Description** : utilise l’algorithme de Canny pour détecter les bords (après conversion en niveaux de gris).
- **But** : met en évidence les contours, utile pour détecter les bordures des lésions sur les feuilles.

### ➤ 3. ROI (Region Of Interest)
- **Description** : masque l’image en conservant uniquement une zone rectangulaire centrale.
- **But** : se concentrer sur la région centrale de la feuille où les symptômes sont souvent localisés.

### ➤ 4. Color Histogram (histogramme de couleurs)
- **Description** : génère un histogramme des canaux RGB montrant la répartition des intensités.
- **But** : analyser la dominance de certaines couleurs, qui peuvent être symptomatiques (jaunissement, brunissement...)

### ➤ 5. Pseudolandmarks (points clés fictifs)
- **Description** : place des points fictifs sur l’image (aux 3 coins d’un triangle) pour illustrer des points d’intérêt.
- **But** : aide à conceptualiser la détection de points caractéristiques pour l’étude des formes.

### ➤ 6. Binary Mask (masque binaire)
- **Description** : transforme l’image en noir et blanc selon un seuil d’intensité (binarisation).
- **But** : isole la forme de la feuille ou met en évidence les lésions.

#### ➤ Commande :
```
make run-transformation-balanced
```
- Gaussian Blur, Edge Detection, ROI, Color Histogram, etc.

### 4️⃣ Entraînement du modèle CNN : **Créer l'intelligence**

#### ➤ Qu'est-ce qu'un CNN ?
Un **Convolutional Neural Network (CNN)** est un type de réseau de neurones adapté à l'analyse d'images.  
Il permet de **détecter automatiquement des caractéristiques visuelles** (bords, motifs, textures).

#### ➤ Fonctionnement général :
1. **Convolution** : filtre l’image pour extraire des **motifs visuels locaux**.  
2. **Pooling** : réduit la taille des données tout en gardant les **informations essentielles**.  
3. **Flatten et Dense** : transforme ces données en **vecteurs** exploitables pour la classification.  
4. **Softmax** : fournit la probabilité pour chaque classe de maladie.

#### ➤ Entraînement :
- Les images passent dans le CNN ➜ il prédit la classe ➜ compare avec la vérité terrain ➜ ajuste ses paramètres (poids).  
- Ce processus se répète sur **plusieurs époques** (cycles) pour **minimiser l'erreur** et **améliorer la précision**.

#### ➤ Commande :
```
make train
```
- CNN personnalisé.
- Résultat : modèle H5, dataset.zip + signature.txt, courbes PNG.

### 5️⃣ Prédictions sur nouvelles images : **Tester le modèle**

### ➤ Qu'est-ce qu'une prédiction ?

Une **prédiction** est le résultat du **modèle CNN** lorsqu’on lui fournit une nouvelle **image inconnue**.  
Le modèle retourne une **probabilité** pour chaque classe possible (exemple : Apple_healthy, Grape_Esca, etc.).

- La **classe ayant la probabilité la plus élevée** est considérée comme la **classe prédite**.  
- La prédiction fournit aussi une **deuxième meilleure hypothèse** (Top 2), ce qui peut être utile pour le **diagnostic**.

---

### ➤ Fonctionnement du processus de prédiction

1. **Chargement et préparation de l'image**  
   - Conversion en RGB.  
   - Redimensionnement à 224x224 pixels.  
   - Normalisation des pixels (valeurs entre 0 et 1).

2. **Passage dans le modèle CNN**  
   - Le modèle génère un **vecteur de probabilités**, chaque valeur représentant la probabilité que l'image appartienne à une classe.

3. **Identification de la classe prédite (Top 1)**  
   - La classe ayant la **probabilité maximale** est retournée comme prédiction principale.  
   - Exemple : 98% de probabilité pour "Apple_healthy".

4. **Affichage de la seconde meilleure prédiction (Top 2)**  
   - Affiche la **deuxième classe probable**, ce qui peut aider à comprendre les erreurs du modèle.

---

### ➤ Exemple de sortie dans le terminal

```
Apple_healthy_1.JPG ➜ ✅ Apple_healthy (98.45%) (2nd: Apple_rust: 1.23%)
```

- ✅ indique que la **classe prédite correspond à la vraie classe** (validation correcte).  
- Le **Top 2** donne une indication de la seconde meilleure estimation (utile si les classes sont proches).

---

### ➤ Pourquoi utiliser une matrice de confusion ?

Une **matrice de confusion** permet de **visualiser la performance globale** du modèle sur un ensemble d'images de test.

#### But :
- Comparer les **classes réelles** et les **classes prédites**.  
- Identifier les **erreurs fréquentes** (exemple : Grape_Black_rot confondu avec Grape_spot).  
- Évaluer si certaines classes sont **sur-prédictées** ou **ignorées**.

#### Lecture de la matrice :
- **Diagonale principale** ➜ prédictions correctes.  
- **Hors diagonale** ➜ erreurs de classification.  
- Une bonne matrice de confusion est **proche d'une diagonale pure**.

---

### ➤ Commandes pour lancer les prédictions sur dossiers

```
make predict-folder-unit1
make predict-folder-unit2
```

- Prédit toutes les images contenues dans le dossier `Unit_test1` ou `Unit_test2`.  
- Génère des **fichiers de résultats** et des **rapports visuels**.

---

### ➤ Fichiers générés dans le dossier output/predictions

- `predictions_<dossier>.txt` ➜ Liste de toutes les prédictions, avec scores de confiance.  
- `errors_<dossier>.txt` ➜ Liste des erreurs de prédiction, pour chaque image :  
  - Classe réelle.  
  - Classe prédite.  
- `confusion_matrix_<dossier>.png` ➜ Matrice de confusion, visuelle et sauvegardée au format image.

---

### ➤ Comment interpréter les résultats

- Une **précision globale élevée** (>90%) signifie un modèle performant.  
- La **matrice de confusion** doit être **bien diagonale**, montrant peu ou pas d'erreurs.  
- Le **fichier errors.txt** permet d'analyser précisément quelles images sont **mal classées**, et **pourquoi**.

---

## 🔬 Pourquoi ces analyses sont importantes ?

- Cela permet de **valider** le modèle avant déploiement.  
- Identifier les **faiblesses spécifiques** (exemple : difficulté à distinguer deux maladies proches).  
- Permet de **réentraîner** le modèle ou d'ajuster l'augmentation si certaines classes sont difficiles à classifier.

---

## 📂 Exemple simplifié d'une matrice de confusion

|                   | Pred: Healthy | Pred: Scab | Pred: Rust |
|-------------------|:-------------:|:----------:|:---------:|
| **True: Healthy** | 50            |  2         | 1         |
| **True: Scab**    | 3             | 45         | 5         |
| **True: Rust**    | 2             | 4          | 46        |

Ici :  
- Les prédictions sont **majoritairement correctes** sur la diagonale.  
- Quelques erreurs entre Scab et Rust ➔ **peut suggérer une similitude visuelle** dans les symptômes.

---

## ⚙️ Fonctionnement Docker

```
make build
make shell
```

---

## 🔎 Tests et Validation

```
make test
make test-flake8
```

- Tests unitaires : pytest.
- Norme : flake8 sur tout le projet.

---

## 📂 Sources / Documentation

- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Matplotlib](https://matplotlib.org/stable/contents.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Docker](https://docs.docker.com/)
- [flake8](https://flake8.pycqa.org/en/latest/)

---
