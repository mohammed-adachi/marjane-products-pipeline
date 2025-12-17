import pandas as pd
import re
from collections import Counter
import json
from scrap import scrape_data_with_selenium

def extract_price_value(price_str):
    """Extrait la valeur numérique du prix"""
    if not price_str:
        return None
    # Extraire les nombres avec virgules/points
    match = re.search(r'(\d+(?:[,\.]\d+)?)', price_str)
    if match:
        return float(match.group(1).replace(',', '.'))
    return None

def categorize_product(title):
    """Catégorise le produit selon son titre"""
    title_lower = title.lower()
    
    categories = {
        'Électronique': ['téléviseur', 'tv', 'écran', 'hisense', 'samsung', 'lg', 'électroménager'],
        'Alimentaire': ['chocolat', 'biscuit', 'lait', 'eau', 'jus', 'fromage', 'yaourt', 'huile', 'tomate'],
        'Hygiène & Beauté': ['shampoing', 'savon', 'crème', 'déodorant', 'dentifrice', 'parfum'],
        'Maison': ['lessive', 'assouplissant', 'nettoyage', 'détergent', 'fairy', 'tide'],
        'Sport & Supporters': ['drapeau', 'vuvuzela', 'mug', 'can 2025', 'maroc'],
        'Fêtes': ['bûche', 'chocolat', 'calendrier', 'bonbon', 'cadeau']
    }
    
    for category, keywords in categories.items():
        if any(keyword in title_lower for keyword in keywords):
            return category
    
    return 'Autre'

def extract_brand(title):
    """Extrait la marque du produit"""
    # Chercher le pattern " - MARQUE" à la fin du titre
    match = re.search(r'-\s*([A-Z][A-Z\s&]+)$', title)
    if match:
        return match.group(1).strip()
    return 'Non spécifié'

def analyze_data(articles):
    """Analyse les données des produits"""
    if not articles:
        print("Aucune donnée à analyser")
        return
    
    # Créer un DataFrame
    df = pd.DataFrame(articles)
    
    # Extraire les valeurs numériques des prix
    df['prix_numerique'] = df['price'].apply(extract_price_value)
    
    # Catégoriser les produits
    df['categorie'] = df['title'].apply(categorize_product)
    
    # Extraire les marques
    df['marque'] = df['title'].apply(extract_brand)
    
    # Filtrer les produits avec prix valides
    df_with_price = df[df['prix_numerique'].notna()]
    
    print("="*80)
    print("📊 ANALYSE DES DONNÉES - MARJANE.MA")
    print("="*80)
    
    # Statistiques générales
    print(f"\n✅ Nombre total de produits extraits: {len(df)}")
    print(f"💰 Produits avec prix: {len(df_with_price)}")
    print(f"🖼️  Produits avec images: {df['image'].notna().sum()}")
    
    # Statistiques de prix
    if len(df_with_price) > 0:
        print("\n" + "="*80)
        print("💵 STATISTIQUES DES PRIX")
        print("="*80)
        print(f"Prix moyen: {df_with_price['prix_numerique'].mean():.2f} DH")
        print(f"Prix médian: {df_with_price['prix_numerique'].median():.2f} DH")
        print(f"Prix minimum: {df_with_price['prix_numerique'].min():.2f} DH")
        print(f"Prix maximum: {df_with_price['prix_numerique'].max():.2f} DH")
        print(f"Écart-type: {df_with_price['prix_numerique'].std():.2f} DH")
        
        # Produits les plus chers
        print("\n🔝 TOP 5 PRODUITS LES PLUS CHERS:")
        top_expensive = df_with_price.nlargest(5, 'prix_numerique')[['title', 'prix_numerique']]
        for idx, row in top_expensive.iterrows():
            print(f"  • {row['title'][:60]}... - {row['prix_numerique']:.2f} DH")
        
        # Produits les moins chers
        print("\n💡 TOP 5 PRODUITS LES MOINS CHERS:")
        top_cheap = df_with_price.nsmallest(5, 'prix_numerique')[['title', 'prix_numerique']]
        for idx, row in top_cheap.iterrows():
            print(f"  • {row['title'][:60]}... - {row['prix_numerique']:.2f} DH")
    
    # Analyse par catégorie
    print("\n" + "="*80)
    print("📦 RÉPARTITION PAR CATÉGORIE")
    print("="*80)
    category_counts = df['categorie'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{category:20s}: {count:3d} produits ({percentage:.1f}%)")
    
    # Prix moyen par catégorie
    if len(df_with_price) > 0:
        print("\n💰 PRIX MOYEN PAR CATÉGORIE:")
        category_avg = df_with_price.groupby('categorie')['prix_numerique'].agg(['mean', 'count'])
        category_avg = category_avg.sort_values('mean', ascending=False)
        for category, row in category_avg.iterrows():
            if row['count'] > 0:
                print(f"{category:20s}: {row['mean']:7.2f} DH (basé sur {int(row['count'])} produits)")
    
    # Analyse des marques
    print("\n" + "="*80)
    print("🏷️  TOP 10 MARQUES LES PLUS PRÉSENTES")
    print("="*80)
    brand_counts = df['marque'].value_counts().head(10)
    for brand, count in brand_counts.items():
        if brand != 'Non spécifié':
            percentage = (count / len(df)) * 100
            print(f"{brand:25s}: {count:3d} produits ({percentage:.1f}%)")
    
    # Détection des promotions
    print("\n" + "="*80)
    print("🎁 DÉTECTION DES PROMOTIONS")
    print("="*80)
    promo_keywords = ['remise', 'promotion', '-', '%', 'achetés']
    df['has_promo'] = df['price'].str.lower().str.contains('|'.join(promo_keywords), na=False)
    promo_count = df['has_promo'].sum()
    print(f"Produits en promotion détectés: {promo_count} ({(promo_count/len(df)*100):.1f}%)")
    
    if promo_count > 0:
        print("\n🔥 QUELQUES PRODUITS EN PROMOTION:")
        promo_products = df[df['has_promo']].head(5)
        for idx, row in promo_products.iterrows():
            print(f"  • {row['title'][:60]}")
            print(f"    Prix: {row['price'][:80]}")
    
    # Analyse des mots-clés dans les titres
    print("\n" + "="*80)
    print("🔍 MOTS-CLÉS LES PLUS FRÉQUENTS DANS LES TITRES")
    print("="*80)
    # Extraire tous les mots des titres
    all_words = []
    stop_words = {'de', 'la', 'le', 'les', 'et', 'en', 'au', 'du', 'à', 'pour', 'avec', 'x', '-'}
    for title in df['title']:
        words = re.findall(r'\b[a-zàâäéèêëïîôùûüç]{3,}\b', title.lower())
        all_words.extend([w for w in words if w not in stop_words])
    
    word_freq = Counter(all_words).most_common(15)
    for word, count in word_freq:
        print(f"{word:15s}: {count:3d} occurrences")
    
    # Sauvegarder les résultats
    print("\n" + "="*80)
    print("💾 SAUVEGARDE DES DONNÉES")
    print("="*80)
    
    # Sauvegarder en CSV
    df.to_csv('produits_marjane.csv', index=False, encoding='utf-8')
    print("✅ Données sauvegardées dans 'produits_marjane.csv'")
    
    # Sauvegarder en JSON
    with open('produits_marjane.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print("✅ Données sauvegardées dans 'produits_marjane.json'")
    
    # Créer un rapport d'analyse
    report = {
        'nombre_produits': len(df),
        'produits_avec_prix': len(df_with_price),
        'prix_moyen': float(df_with_price['prix_numerique'].mean()) if len(df_with_price) > 0 else 0,
        'prix_min': float(df_with_price['prix_numerique'].min()) if len(df_with_price) > 0 else 0,
        'prix_max': float(df_with_price['prix_numerique'].max()) if len(df_with_price) > 0 else 0,
        'categories': category_counts.to_dict(),
        'top_marques': brand_counts.head(5).to_dict(),
        'produits_en_promotion': int(promo_count)
    }
    
    with open('analyse_rapport.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("✅ Rapport d'analyse sauvegardé dans 'analyse_rapport.json'")
    
    print("\n" + "="*80)
    print("✨ ANALYSE TERMINÉE")
    print("="*80)
    
    return df

def main():
    """Fonction principale"""
    print("🚀 Démarrage du scraping de Marjane.ma...")
    
    # Scraper les données
    articles = scrape_data_with_selenium('https://www.marjane.ma/')
    
    # Analyser les données
    if articles:
        analyze_data(articles)
    else:
        print("❌ Aucune donnée n'a pu être extraite.")

if __name__ == "__main__":
    main()
