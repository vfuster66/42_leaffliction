
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

**Commandes :**
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

```
make run-transformation-balanced
```
- Gaussian Blur, Edge Detection, ROI, Color Histogram, etc.

### 4️⃣ Entraînement du modèle CNN : **Créer l'intelligence**
```
make train
```
- CNN personnalisé.
- Résultat : modèle H5, dataset.zip + signature.txt, courbes PNG.

### 5️⃣ Prédictions sur nouvelles images : **Tester le modèle**
```
make predict-folder-unit1
make predict-folder-unit2
```
- Top 2 classes + probabilités.
- Matrice de confusion + logs d'erreurs.

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

## 📖 Explication du modèle CNN

- 3 Conv2D + MaxPooling.
- Flatten + Dense (256) + Dropout (0.5).
- Sortie Softmax (classification).

**Pourquoi ?**
- Léger, rapide (Mac M1).
- Suffisant pour un dataset modeste.

---

## 📂 Sources / Documentation

- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Matplotlib](https://matplotlib.org/stable/contents.html)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Docker](https://docs.docker.com/)
- [flake8](https://flake8.pycqa.org/en/latest/)

---
