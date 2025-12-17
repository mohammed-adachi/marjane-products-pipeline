import pandas as pd
import json
import re
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Classe pour le nettoyage et la structuration des données de Marjane"""
    
    def __init__(self, json_file='produits_marjane.json'):
        """Initialise l'analyseur avec les données"""
        self.json_file = json_file
        self.df = None
        self.data_cleaned = None
        self.load_data()
    
    def load_data(self):
        """Charge les données depuis le fichier JSON"""
        print("📂 Chargement des données...")
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
            print(f"✅ {len(self.df)} produits chargés avec succès\n")
        except FileNotFoundError:
            print(f"❌ Fichier {self.json_file} non trouvé")
            return False
        return True
    
    def clean_data(self):
        """Nettoie et structure les données"""
        print("="*80)
        print("🧹 NETTOYAGE ET STRUCTURATION DES DONNÉES")
        print("="*80)
        
        # Créer une copie pour les données nettoyées
        df = self.df.copy()
        
        print(f"\n📊 État initial des données:")
        print(f"   Nombre de lignes: {len(df)}")
        print(f"   Nombre de colonnes: {len(df.columns)}")
        print(f"   Colonnes: {list(df.columns)}")
        
        # 1. NETTOYAGE DES TITRES
        print(f"\n{'─'*80}")
        print("1️⃣  NETTOYAGE DES TITRES")
        print(f"{'─'*80}")
        
        df['title_original'] = df['title']
        # Supprimer les espaces inutiles
        df['title'] = df['title'].str.strip()
        # Normaliser les espaces multiples
        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)
        
        # Détacter les doublons
        duplicates = df.duplicated(subset=['title']).sum()
        print(f"✅ Espaces nettoyés")
        print(f"⚠️  Doublons détectés: {duplicates}")
        
        # 2. NETTOYAGE DES PRIX
        print(f"\n{'─'*80}")
        print("2️⃣  NETTOYAGE DES PRIX")
        print(f"{'─'*80}")
        
        df['price_original'] = df['price']
        
        # Extraire le prix principal
        df['prix_principal'] = df['price'].apply(self.extract_main_price)
        
        # Extraire le prix réduit (s'il existe)
        df['prix_reduit'] = df['price'].apply(self.extract_reduced_price)
        
        # Calculer la remise en pourcentage
        df['pourcentage_remise'] = df.apply(self.calculate_discount, axis=1)
        
        # Vérifier les prix valides
        valid_prices = df['prix_principal'].notna().sum()
        print(f"✅ Prix principal extrait: {valid_prices}/{len(df)}")
        print(f"✅ Prix réduits détectés: {df['prix_reduit'].notna().sum()}")
        print(f"✅ Remises calculées: {df['pourcentage_remise'].notna().sum()}")
        
        # 3. NETTOYAGE DES IMAGES
        print(f"\n{'─'*80}")
        print("3️⃣  NETTOYAGE DES URLS D'IMAGES")
        print(f"{'─'*80}")
        
        df['image_original'] = df['image']
        
        # Vérifier la validité des URLs d'images
        df['image_valide'] = df['image'].apply(self.validate_image_url)
        
        # Extraire le domaine de l'image
        df['image_domaine'] = df['image'].apply(self.extract_domain)
        
        image_count = df['image_valide'].sum()
        print(f"✅ URLs d'images valides: {image_count}/{len(df)}")
        print(f"✅ Domaines: {df['image_domaine'].unique().tolist()}")
        
        # 4. EXTRACTION DES MÉTADONNÉES
        print(f"\n{'─'*80}")
        print("4️⃣  EXTRACTION DES MÉTADONNÉES")
        print(f"{'─'*80}")
        
        # Extraire la marque
        df['marque'] = df['title'].apply(self.extract_brand)
        
        # Extraire la catégorie
        df['categorie'] = df['title'].apply(self.categorize_product)
        
        # Extraire la taille/quantité
        df['taille_quantite'] = df['title'].apply(self.extract_size)
        
        # Détecter les promotions
        df['en_promotion'] = df['price'].str.lower().str.contains(
            'remise|promotion|-|%|achetés|offre', 
            na=False, 
            regex=True
        )
        
        # Extraire le type de promotion
        df['type_promotion'] = df['price'].apply(self.extract_promotion_type)
        
        print(f"✅ Marques extraites: {df['marque'].nunique()} uniques")
        print(f"✅ Catégories détectées: {df['categorie'].nunique()} uniques")
        print(f"✅ Produits en promotion: {df['en_promotion'].sum()}")
        
        # 5. VALIDATION ET COMPLÉTUDE
        print(f"\n{'─'*80}")
        print("5️⃣  VALIDATION ET COMPLÉTUDE DES DONNÉES")
        print(f"{'─'*80}")
        
        # Taux de complétude par colonne
        print("\n📋 Complétude des données (%):")
        for col in ['title', 'price', 'image', 'marque', 'categorie']:
            completeness = (df[col].notna().sum() / len(df)) * 100
            status = "✅" if completeness == 100 else "⚠️ " if completeness > 80 else "❌"
            print(f"   {status} {col:20s}: {completeness:6.1f}%")
        
        # 6. STATISTIQUES DE QUALITÉ
        print(f"\n{'─'*80}")
        print("6️⃣  STATISTIQUES DE QUALITÉ")
        print(f"{'─'*80}")
        
        # Longueur des titres
        df['longueur_titre'] = df['title'].str.len()
        print(f"\n📝 Longueur des titres:")
        print(f"   Min: {df['longueur_titre'].min()} caractères")
        print(f"   Max: {df['longueur_titre'].max()} caractères")
        print(f"   Moyenne: {df['longueur_titre'].mean():.1f} caractères")
        
        # Distribution des prix valides
        prix_valides = df[df['prix_principal'].notna()]['prix_principal']
        if len(prix_valides) > 0:
            print(f"\n💰 Distribution des prix (pour {len(prix_valides)} produits):")
            print(f"   Min: {prix_valides.min():.2f} DH")
            print(f"   Q1: {prix_valides.quantile(0.25):.2f} DH")
            print(f"   Médiane: {prix_valides.median():.2f} DH")
            print(f"   Q3: {prix_valides.quantile(0.75):.2f} DH")
            print(f"   Max: {prix_valides.max():.2f} DH")
        
        # 7. DÉTECTION DES ANOMALIES
        print(f"\n{'─'*80}")
        print("7️⃣  DÉTECTION DES ANOMALIES")
        print(f"{'─'*80}")
        
        anomalies = 0
        
        # Titres vides
        titres_vides = df['title'].isna().sum()
        if titres_vides > 0:
            print(f"⚠️  Titres vides: {titres_vides}")
            anomalies += titres_vides
        
        # Images vides
        images_vides = df['image'].isna().sum()
        if images_vides > 0:
            print(f"⚠️  Images manquantes: {images_vides}")
            anomalies += images_vides
        
        # Titres trop courts
        titres_courts = (df['longueur_titre'] < 5).sum()
        if titres_courts > 0:
            print(f"⚠️  Titres trop courts (<5 caractères): {titres_courts}")
            anomalies += titres_courts
        
        # Prix invalides
        prix_invalides = df['prix_principal'].isna().sum()
        if prix_invalides > 0:
            print(f"⚠️  Prix invalides: {prix_invalides}")
            anomalies += prix_invalides
        
        # Doublons
        if duplicates > 0:
            print(f"⚠️  Doublons de titre: {duplicates}")
            anomalies += duplicates
        
        if anomalies == 0:
            print("✅ Aucune anomalie détectée!")
        
        self.data_cleaned = df
        return df
    
    def extract_main_price(self, price_str):
        """Extrait le prix principal"""
        if not price_str or not isinstance(price_str, str):
            return None
        
        # Chercher le premier nombre avec DH
        match = re.search(r'(\d+(?:[,\.]\d+)?)\s*DH', price_str, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', '.'))
        
        # Chercher juste un nombre
        match = re.search(r'(\d+(?:[,\.]\d+)?)', price_str)
        if match:
            return float(match.group(1).replace(',', '.'))
        
        return None
    
    def extract_reduced_price(self, price_str):
        """Extrait le prix réduit (s'il existe)"""
        if not price_str or not isinstance(price_str, str):
            return None
        
        # Chercher les nombres séparés par un trait ou autre
        matches = re.findall(r'(\d+(?:[,\.]\d+)?)\s*DH', price_str, re.IGNORECASE)
        
        if len(matches) >= 2:
            return float(matches[-1].replace(',', '.'))
        
        return None
    
    def calculate_discount(self, row):
        """Calcule le pourcentage de remise"""
        if pd.isna(row['prix_principal']) or pd.isna(row['prix_reduit']):
            return None
        
        if row['prix_principal'] == 0:
            return None
        
        discount = ((row['prix_principal'] - row['prix_reduit']) / row['prix_principal']) * 100
        return max(0, min(100, discount))  # Entre 0 et 100%
    
    def validate_image_url(self, url):
        """Valide une URL d'image"""
        if not url or not isinstance(url, str):
            return False
        
        valid_domains = ['cloudinary.com', 'marjane.ma', 'res.cloudinary.com']
        return any(domain in url.lower() for domain in valid_domains)
    
    def extract_domain(self, url):
        """Extrait le domaine de l'image"""
        if not url or not isinstance(url, str):
            return 'Inconnu'
        
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            return match.group(1)
        
        return 'Inconnu'
    
    def extract_brand(self, title):
        """Extrait la marque du titre"""
        if not title or not isinstance(title, str):
            return 'Non spécifié'
        
        # Pattern: " - MARQUE" à la fin
        match = re.search(r'-\s*([A-Z][A-Z\s&\']+)$', title)
        if match:
            return match.group(1).strip()
        
        return 'Non spécifié'
    
    def categorize_product(self, title):
        """Catégorise le produit"""
        if not title or not isinstance(title, str):
            return 'Autre'
        
        title_lower = title.lower()
        
        categories = {
            'Électronique': ['téléviseur', 'tv', 'écran', 'hisense', 'samsung', 'lg'],
            'Alimentaire': ['chocolat', 'biscuit', 'lait', 'eau', 'jus', 'huile', 'tomate', 'safran'],
            'Hygiène & Beauté': ['shampoing', 'savon', 'crème', 'déodorant', 'dentifrice'],
            'Maison & Nettoyage': ['lessive', 'assouplissant', 'nettoyage', 'détergent', 'fairy'],
            'Sport & Supporters': ['drapeau', 'vuvuzela', 'mug', 'can', 'maroc'],
            'Fêtes & Occasions': ['bûche', 'calendrier', 'bonbon', 'cadeau']
        }
        
        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        
        return 'Autre'
    
    def extract_size(self, title):
        """Extrait la taille/quantité du produit"""
        if not title or not isinstance(title, str):
            return 'Non spécifié'
        
        # Chercher les patterns de taille
        patterns = [
            r'(\d+\s*(?:ml|l|g|kg|cm|pouces|x))',
            r'(\d+\s*(?:pièces|pieces|pack))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return 'Non spécifié'
    
    def extract_promotion_type(self, price_str):
        """Extrait le type de promotion"""
        if not price_str or not isinstance(price_str, str):
            return 'Aucune'
        
        price_lower = price_str.lower()
        
        if 'remise' in price_lower or '-' in price_str:
            return 'Remise'
        elif '%' in price_str:
            return 'Pourcentage'
        elif 'achetés' in price_lower:
            return 'Promotion multi-achat'
        elif 'offre' in price_lower:
            return 'Offre spéciale'
        
        return 'Aucune'
    
    def save_cleaned_data(self):
        """Sauvegarde les données nettoyées"""
        print(f"\n{'─'*80}")
        print("💾 SAUVEGARDE DES DONNÉES NETTOYÉES")
        print(f"{'─'*80}\n")
        
        if self.data_cleaned is None:
            print("❌ Aucune donnée nettoyée à sauvegarder")
            return
        
        # Sauvegarder en CSV
        csv_file = 'produits_marjane_clean.csv'
        self.data_cleaned.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✅ Données nettoyées sauvegardées: {csv_file}")
        
        # Sauvegarder en JSON avec structure améliorée
        json_file = 'produits_marjane_analyse.json'
        
        # Préparer les données pour JSON
        data_json = []
        for idx, row in self.data_cleaned.iterrows():
            item = {
                'id': idx + 1,
                'titre': row['title'],
                'prix': {
                    'principal': float(row['prix_principal']) if pd.notna(row['prix_principal']) else None,
                    'reduit': float(row['prix_reduit']) if pd.notna(row['prix_reduit']) else None,
                    'remise_pourcentage': float(row['pourcentage_remise']) if pd.notna(row['pourcentage_remise']) else None,
                    'devise': 'DH'
                },
                'image': {
                    'url': row['image'],
                    'valide': bool(row['image_valide']),
                    'domaine': row['image_domaine']
                },
                'metadata': {
                    'marque': row['marque'],
                    'categorie': row['categorie'],
                    'taille_quantite': row['taille_quantite'],
                    'en_promotion': bool(row['en_promotion']),
                    'type_promotion': row['type_promotion'],
                    'longueur_titre': int(row['longueur_titre'])
                }
            }
            data_json.append(item)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, ensure_ascii=False, indent=2)
        print(f"✅ Données structurées sauvegardées: {json_file}")
        
        # Créer un rapport de nettoyage
        report = {
            'timestamp': datetime.now().isoformat(),
            'nombre_produits_traites': len(self.data_cleaned),
            'taux_complétude_moyen': float(self.data_cleaned.notna().sum().sum() / (len(self.data_cleaned) * len(self.data_cleaned.columns)) * 100),
            'colonnes_créées': [
                'prix_principal', 'prix_reduit', 'pourcentage_remise',
                'image_valide', 'image_domaine', 'marque', 'categorie',
                'taille_quantite', 'en_promotion', 'type_promotion', 'longueur_titre'
            ],
            'statistiques': {
                'prix_moyen': float(self.data_cleaned['prix_principal'].mean()) if self.data_cleaned['prix_principal'].notna().sum() > 0 else None,
                'produits_en_promotion': int(self.data_cleaned['en_promotion'].sum()),
                'images_valides': int(self.data_cleaned['image_valide'].sum()),
                'marques_uniques': int(self.data_cleaned['marque'].nunique()),
                'categories': self.data_cleaned['categorie'].value_counts().to_dict()
            }
        }
        
        report_file = 'rapport_nettoyage.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ Rapport de nettoyage sauvegardé: {report_file}")
    
    def display_sample(self, n=5):
        """Affiche un échantillon des données nettoyées"""
        if self.data_cleaned is None:
            print("❌ Aucune donnée nettoyée à afficher")
            return
        
        print(f"\n{'─'*80}")
        print(f"📋 ÉCHANTILLON DES DONNÉES NETTOYÉES ({n} produits)")
        print(f"{'─'*80}\n")
        
        for idx, row in self.data_cleaned.head(n).iterrows():
            print(f"🔹 Produit {idx + 1}")
            print(f"   Titre: {row['title']}")
            print(f"   Marque: {row['marque']}")
            print(f"   Catégorie: {row['categorie']}")
            print(f"   Prix: {row['prix_principal']:.2f} DH" if pd.notna(row['prix_principal']) else "   Prix: N/A")
            if pd.notna(row['prix_reduit']):
                print(f"   Prix réduit: {row['prix_reduit']:.2f} DH (-{row['pourcentage_remise']:.1f}%)")
            print(f"   Taille: {row['taille_quantite']}")
            print(f"   Promotion: {'✅ ' + row['type_promotion'] if row['en_promotion'] else '❌ Non'}")
            print()

def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("🔍 DATA ANALYSIS - NETTOYAGE & STRUCTURATION")
    print("="*80 + "\n")
    
    # Créer l'analyseur
    analyzer = DataAnalyzer()
    
    if analyzer.df is None:
        print("❌ Impossible de charger les données")
        return
    
    # Nettoyer les données
    analyzer.clean_data()
    
    # Afficher un échantillon
    analyzer.display_sample(n=5)
    
    # Sauvegarder les données nettoyées
    analyzer.save_cleaned_data()
    
    print("\n" + "="*80)
    print("✨ ANALYSE COMPLÉTÉE AVEC SUCCÈS")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
