import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_data_with_selenium(url):
    """Scraping avec Selenium pour les sites JavaScript"""
    options = Options()
    options.add_argument('--headless')  # Mode sans interface graphique
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.get(url)
        
        # Attendre plus longtemps pour Cloudflare
        print("Attente du chargement de la page...")
        time.sleep(10)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Sauvegarder le HTML pour déboguer
        with open('page_source.html', 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print("HTML sauvegardé dans page_source.html")
        
        driver.quit()
        
        articles = []
        
        # Méthode 1: Chercher les produits dans des divs ou liens qui contiennent des éléments de produit typiques
        for item in soup.find_all(['div', 'a'], class_=lambda x: x and any(word in str(x).lower() for word in ['product', 'card', 'item'])):
            # Ignorer les éléments du menu et de navigation
            if item.find_parent(['nav', 'header', 'footer']):
                continue
                
            title_elem = item.find(['h2', 'h3', 'h4', 'span'], class_=lambda x: x and any(word in str(x).lower() for word in ['title', 'name', 'label']))
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Ne garder que les titres valides (pas trop courts, pas trop longs)
            if not title or len(title) < 10 or len(title) > 200:
                continue
            
            # Chercher le prix
            price_elem = item.find(['span', 'div', 'p'], class_=lambda x: x and 'price' in str(x).lower())
            price = price_elem.get_text(strip=True) if price_elem else ""
            
            # Chercher l'image
            img = item.find('img')
            image_url = ""
            if img:
                image_url = img.get('src', '') or img.get('data-src', '') or img.get('data-lazy-src', '')
            
            # Ne garder que les éléments qui ont un prix ou une image de produit valide
            if (price and 'DH' in price.upper()) or (image_url and not any(x in image_url.lower() for x in ['logo', 'icon', 'svg'])):
                articles.append({
                    'title': title,
                    'price': price,
                    'image': image_url
                })
        
        return articles
    except Exception as e:
        print(f"Erreur Selenium: {e}")
        import traceback
        traceback.print_exc()
        return []

def scrape_data(url):
    """Scraping simple avec requests (si le site n'utilise pas JavaScript)"""
    response = requests.get(url)
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup.prettify()[:500])  # Pour aider à inspecter la structure HTML
    articles = []
    
    # Sélecteurs adaptés pour Marjane.ma
    # Les produits sont généralement dans des divs ou sections spécifiques
    for item in soup.find_all(['div', 'a'], class_=lambda x: x and ('product' in x.lower() or 'item' in x.lower())):
        title_elem = item.find(['h2', 'h3', 'h4', 'span', 'p'], class_=lambda x: x and ('title' in x.lower() or 'name' in x.lower()))
        title = title_elem.text.strip() if title_elem else item.get('title', '')
        
        # Chercher le prix
        price_elem = item.find(['span', 'div', 'p'], class_=lambda x: x and 'price' in x.lower())
        price = price_elem.text.strip() if price_elem else ""
        
        # Chercher l'image
        img = item.find('img')
        image_url = img.get('src', '') if img else ""
        
        if title:
            articles.append({
                'title': title,
                'price': price,
                'image': image_url
            })
            
    return articles

# Test
data = scrape_data_with_selenium('https://www.marjane.ma/')
print(f"\nNombre d'articles trouvés: {len(data)}")
for i, article in enumerate(data[:5], 1):  # Afficher les 5 premiers
    print(f"\n{i}. {article['title']}")
    if article.get('price'):
        print(f"   Prix: {article['price']}")
    if article.get('image'):
        print(f"   Image: {article['image'][:50]}...")