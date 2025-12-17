import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import List, Dict
import re
import warnings
warnings.filterwarnings('ignore')

class SemanticSearcher:
    """Classe pour effectuer une recherche sémantique avec Scikit-learn et TF-IDF"""
    
    def __init__(self):
        """Initialise le searcher avec TF-IDF"""
        print(f"📦 Initialisation du moteur de recherche sémantique")
        print("   Utilisant TF-IDF + Similarité Cosinus\n")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        self.products = []
        self.embeddings = None
        self.df = None
        self.titles = []
        
        print(f"✅ Moteur de recherche initialisé avec succès\n")
    
    def load_products(self, json_file='produits_marjane.json'):
        """Charge les produits depuis un fichier JSON"""
        print(f"📂 Chargement des produits depuis {json_file}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
            
            self.titles = [p.get('title', '') for p in self.products]
            print(f"✅ {len(self.products)} produits chargés\n")
            return True
        except FileNotFoundError:
            print(f"❌ Fichier {json_file} non trouvé\n")
            return False
    
    def preprocess_text(self, text):
        """Prétraite le texte"""
        if not isinstance(text, str):
            return ""
        
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux sauf les espaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_embeddings(self):
        """Génère les embeddings TF-IDF pour tous les produits"""
        if not self.products:
            print("❌ Aucun produit chargé")
            return False
        
        print("⚙️  Génération des embeddings TF-IDF...")
        print(f"   Total: {len(self.titles)} produits\n")
        
        # Prétraiter les titres
        processed_titles = [self.preprocess_text(title) for title in self.titles]
        
        # Générer les embeddings TF-IDF
        self.embeddings = self.vectorizer.fit_transform(processed_titles).toarray()
        
        print(f"✅ Embeddings générés avec succès")
        print(f"   Forme: {self.embeddings.shape}")
        print(f"   Vocabulaire: {len(self.vectorizer.get_feature_names_out())} termes\n")
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Effectue une recherche sémantique
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
        
        Returns:
            Liste des produits les plus similaires
        """
        if self.embeddings is None:
            print("❌ Les embeddings ne sont pas générés")
            return []
        
        # Prétraiter et vectoriser la requête
        processed_query = self.preprocess_text(query)
        query_embedding = self.vectorizer.transform([processed_query]).toarray()
        
        # Calculer la similarité cosinus
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Obtenir les indices des top-k résultats
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Préparer les résultats
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Ignorer les résultats avec similarité 0
                product = self.products[idx]
                results.append({
                    'index': int(idx),
                    'titre': product.get('title', ''),
                    'prix': product.get('price', ''),
                    'image': product.get('image', ''),
                    'similarite': float(similarities[idx]),
                    'score_pourcentage': float(similarities[idx] * 100)
                })
        
        return results
    
    def display_results(self, results: List[Dict], query: str):
        """Affiche les résultats de la recherche de manière formatée"""
        print(f"\n{'='*90}")
        print(f"🔍 RÉSULTATS POUR: \"{query}\"")
        print(f"{'='*90}\n")
        
        if not results:
            print("❌ Aucun résultat trouvé\n")
            return
        
        for i, result in enumerate(results, 1):
            bar_length = int(result['score_pourcentage'] / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            print(f"🏷️  Résultat {i}")
            print(f"   Titre: {result['titre']}")
            print(f"   Prix: {result['prix']}")
            print(f"   Similarité: [{bar}] {result['score_pourcentage']:.1f}%")
            print()
    
    def interactive_search(self):
        """Mode de recherche interactif"""
        print("\n" + "="*90)
        print("🔎 MODE DE RECHERCHE INTERACTIF")
        print("="*90)
        print("Tapez votre requête ou 'quit' pour quitter\n")
        
        while True:
            query = input("🔍 Votre recherche: ").strip()
            
            if query.lower() == 'quit':
                print("\n👋 Au revoir!")
                break
            
            if not query:
                print("❌ Veuillez entrer une requête valide\n")
                continue
            
            results = self.search(query, top_k=5)
            self.display_results(results, query)
    
    def batch_search(self, queries: List[str], top_k: int = 3) -> Dict:
        """Effectue plusieurs recherches à la fois"""
        print(f"\n{'='*90}")
        print(f"🔍 RECHERCHE PAR BATCH ({len(queries)} requêtes)")
        print(f"{'='*90}\n")
        
        batch_results = {}
        
        for query in queries:
            results = self.search(query, top_k=top_k)
            batch_results[query] = results
            self.display_results(results, query)
        
        return batch_results
    
    def search_by_category(self, query: str, category: str, top_k: int = 5) -> List[Dict]:
        """Recherche limitée à une catégorie spécifique"""
        try:
            df = pd.read_csv('produits_marjane_clean.csv')
        except FileNotFoundError:
            print("❌ Fichier produits_marjane_clean.csv non trouvé")
            return []
        
        category_indices = df[df['categorie'] == category].index.tolist()
        
        if not category_indices:
            print(f"❌ Aucun produit trouvé dans la catégorie: {category}")
            return []
        
        processed_query = self.preprocess_text(query)
        query_embedding = self.vectorizer.transform([processed_query]).toarray()
        
        category_embeddings = self.embeddings[category_indices]
        similarities = cosine_similarity(query_embedding, category_embeddings)[0]
        
        top_local_indices = np.argsort(similarities)[::-1][:top_k]
        top_global_indices = [category_indices[i] for i in top_local_indices]
        
        results = []
        for i, global_idx in enumerate(top_global_indices):
            product = self.products[global_idx]
            results.append({
                'index': int(global_idx),
                'titre': product.get('title', ''),
                'prix': product.get('price', ''),
                'image': product.get('image', ''),
                'similarite': float(similarities[top_local_indices[i]]),
                'score_pourcentage': float(similarities[top_local_indices[i]] * 100)
            })
        
        return results
    
    def similar_products(self, product_index: int, top_k: int = 5) -> List[Dict]:
        """Trouve les produits similaires à un produit donné"""
        if product_index >= len(self.products):
            print(f"❌ Indice de produit invalide: {product_index}")
            return []
        
        product_embedding = self.embeddings[product_index:product_index+1]
        similarities = cosine_similarity(product_embedding, self.embeddings)[0]
        
        similarities[product_index] = -1
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            product = self.products[idx]
            results.append({
                'index': int(idx),
                'titre': product.get('title', ''),
                'prix': product.get('price', ''),
                'image': product.get('image', ''),
                'similarite': float(similarities[idx]),
                'score_pourcentage': float(similarities[idx] * 100)
            })
        
        return results
    
    def export_results(self, results: Dict, filename: str = 'search_results.json'):
        """Exporte les résultats en JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Résultats exportés dans {filename}")

def main():
    """Fonction principale"""
    print("\n" + "="*90)
    print("🚀 SEMANTIC SEARCH AVEC TF-IDF")
    print("="*90 + "\n")
    
    # Initialiser le searcher
    searcher = SemanticSearcher()
    
    # Charger les produits
    if not searcher.load_products():
        return
    
    # Générer les embeddings
    if not searcher.generate_embeddings():
        return
    
    # Exemples de recherches
    print("="*90)
    print("📋 EXEMPLES DE RECHERCHES SÉMANTIQUES")
    print("="*90)
    
    test_queries = [
        "produits de beauté",
        "électronique pas cher",
        "chocolat et bonbons",
        "nettoyage de maison",
        "produits bio"
    ]
    
    batch_results = searcher.batch_search(test_queries, top_k=3)
    
    # Sauvegarder les résultats
    searcher.export_results(batch_results, 'search_results.json')
    
    # Recherche interactive
    print("\n" + "="*90)
    print("✨ MODE INTERACTIF")
    print("="*90)
    print("\nVoulez-vous essayer le mode interactif? (oui/non)")
    response = input("Réponse: ").strip().lower()
    
    if response in ['oui', 'yes', 'o', 'y']:
        searcher.interactive_search()
    else:
        print("\n👋 Merci d'avoir utilisé le Semantic Search!")
    
    print("\n" + "="*90)
    print("✅ PROGRAMME TERMINÉ")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
