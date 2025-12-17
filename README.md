# Marjane Products Pipeline

Un projet complet pour récupérer, analyser et rechercher les produits de **Marjane**.  
Ce pipeline inclut **Web Scraping**, **Data Mining**, **Data Analysis** et **Semantic Search**.

---

## 🛠️ Fonctionnalités

1. **Web Scraping** (`scrap.py`)
   - Récupère les informations des produits depuis le site Marjane
   - Données collectées : nom, prix, catégorie, description, image

2. **Data Mining** (`mining.py`)
   - Nettoie et structure les données
   - Gère les doublons et les données manquantes
   - Enrichit les informations des produits

3. **Data Analysis** (`analyse.py`)
   - Analyse et visualisation des tendances des produits
   - Exemples : distribution des prix, analyse par catégorie, wordcloud des descriptions

4. **Semantic Search** (`Semantic_Search.py`)
   - Recherche intelligente par mot-clé ou phrase
   - Utilise les embeddings pour trouver les produits les plus pertinents

---

## 💻 Installation

1. Cloner le repository :
```bash
git clone https://github.com/mohammed-adachi/marjane-products-pipeline.git
cd marjane-products-pipeline
Créer un environnement virtuel et installer les dépendances :

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
Utilisation

1)Scraping :

python scrap.py


2)Data Mining :

python mining.py


3)Data Analysis :

python analyse.py


4)Semantic Search :

python Semantic_Search.py