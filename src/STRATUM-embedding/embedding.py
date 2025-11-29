import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import google.generativeai as genai
import time

# --- 1. CONFIGURATION GOOGLE ---
# Colle ta clé commençant par AIza ici
GOOGLE_API_KEY = "AIzaSyDSu3vIGYdcWQQnmMCmfwTAJAFjHX554WI"

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. CHARGEMENT EXCEL ---
file_path = "Glossaire.xlsx"
sheet_name = "STRATUM"

print(f"Chargement de '{file_path}'...")
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    # Renommage des colonnes A et B pour être sûr
    df = df.rename(columns={df.columns[0]: 'Code', df.columns[1]: 'Description'})
    df = df[['Code', 'Description']].dropna()
    print(f"{len(df)} lignes chargées.")
    print(df.head(3))
except Exception as e:
    print(f"Erreur chargement Excel : {e}")
    exit()

# --- 3. NETTOYAGE ---
def clean_stratum(text):
    text = str(text)
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text.strip()

df['Cleaned_Text'] = df['Description'].apply(clean_stratum)

# --- 4. EMBEDDING GOOGLE (Optimisé par batch) ---
# Cette fonction envoie les données par paquets de 100 pour aller super vite
def get_google_embeddings(texts):
    embeddings = []
    batch_size = 100
    model = "models/text-embedding-004"
    
    print(f"Génération des embeddings (Google) pour {len(texts)} lignes...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Appel API Google
            result = genai.embed_content(
                model=model,
                content=batch,
                task_type="clustering" # Aide le modèle à comprendre le but
            )
            embeddings.extend(result['embedding'])
            print(f"  > Batch {i} à {i+len(batch)} traité.")
            time.sleep(0.2) # Petite pause de sécurité
        except Exception as e:
            print(f"Erreur sur le batch {i}: {e}")
            # On remplit avec des None pour ne pas casser la structure
            embeddings.extend([None] * len(batch))
            
    return embeddings

# Conversion de la colonne en liste pour l'envoi
text_list = df['Cleaned_Text'].tolist()
vectors = get_google_embeddings(text_list)

df['Embedding'] = vectors
df = df.dropna(subset=['Embedding'])

# --- 5. CLUSTERING K-MEANS ---
print("Calcul du K-Means (5 clusters)...")
matrix = np.vstack(df['Embedding'].values)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(matrix)

# --- 6. EXPORT ---
output_file = "Glossaire_STRATUM_Clustered_Google.xlsx"
# On ne sauvegarde pas la colonne 'Embedding' car elle est illisible dans Excel
df[['Code', 'Description', 'Cleaned_Text', 'Cluster_ID']].to_excel(output_file, index=False)

print(f"\nTerminé ! Fichier généré : {output_file}")