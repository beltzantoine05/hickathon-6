import subprocess
import sys
import os
import time

def run_step(script_name, step_description):
    """
    Exécute un script Python en tant que sous-processus.
    Arrête tout le pipeline si le script échoue.
    """
    print("=" * 60)
    print(f"🚀 DÉMARRAGE : {step_description}")
    print(f"   Script : {script_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # sys.executable assure qu'on utilise le même interpréteur Python (venv/conda)
        # check=True lève une exception si le script retourne une erreur
        subprocess.run([sys.executable, script_name], check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCÈS : {step_description} terminé en {elapsed:.2f}s.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERREUR CRITIQUE lors de {step_description}.")
        print(f"   Le script s'est arrêté avec le code de sortie {e.returncode}.")
        print("   Arrêt du pipeline.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ ERREUR : Le fichier '{script_name}' est introuvable.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE : {e}")
        sys.exit(1)

def check_environment():
    """Vérifications rapides avant de lancer."""
    print("🔍 Vérification de l'environnement...")
    
    # 1. Vérif du .env
    if not os.path.exists(".env"):
        print("⚠️  ATTENTION : Fichier '.env' introuvable.")
        print("   Assure-toi que les variables MINIO_... et WANDB_... sont définies dans le système.")
    else:
        print("   .env trouvé.")

    # 2. Vérif des données
    data_dir = os.path.join("datas") # Selon ton chemin précédent
    if not os.path.exists(data_dir):
         print(f"⚠️  ATTENTION : Le dossier {data_dir} semble absent (vérifie tes chemins).")
    
    print("   Environnement OK.\n")

def main():
    print("\n" + "#" * 40)
    print("   PIPELINE ML : EMBEDDING & REGRESSION")
    print("#" * 40 + "\n")
    
    check_environment()

    # --- ÉTAPE 1 : Denoising Auto-Encoder ---
    # Produit : Scaler, Imputer, Encoder Weights (envoyés sur MinIO/W&B)
    run_step("src/train_dae.py", "1. Pré-entraînement DAE (Unsupervised)")

    print("\n⏳ Pause de 2 secondes pour assurer la propagation des artefacts...\n")
    time.sleep(2)

    # --- ÉTAPE 2 : Supervised Fine-tuning ---
    # Consomme : Les artefacts de l'étape 1 via W&B (:latest)
    # Produit : Modèle final de régression et Embedding final
    run_step("src/train_regressor.py", "2. Fine-tuning Supervisé")

    print("\n" + "=" * 60)
    print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("   Tu peux retrouver tes modèles sur W&B et MinIO.")
    print("=" * 60)

if __name__ == "__main__":
    main()