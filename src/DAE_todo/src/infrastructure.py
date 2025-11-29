import os
import boto3
import wandb
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


class ArtifactManager:
    def __init__(self, bucket_name=None):
        self.bucket = bucket_name or os.getenv("MINIO_BUCKET", "ml-project-artifacts")

        # Client Boto3 pour ton stockage MinIO personnel (Backup souverain)
        # On ne configure plus les variables d'environnement globales pour ne pas perturber W&B
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_ENDPOINT'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
            aws_session_token=os.getenv('MINIO_SESSION_TOKEN', None)
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
                print(f"[MinIO] Bucket '{self.bucket}' created.")
            except Exception:
                pass  # On ignore si déjà créé ou erreur mineure

    def log_model(self, local_path, artifact_name, run, metadata=None, aliases=None):
        """
        1. Upload sur MinIO (Backup perso)
        2. Upload PHYSIQUE sur W&B (Plus robuste, pas de gestion de lien S3)
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        # --- A. BACKUP MINIO (Optionnel mais demandé) ---
        try:
            s3_key = f"models/{run.id}/{artifact_name}.pth"
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            print(f"[MinIO] Backup uploadé : s3://{self.bucket}/{s3_key}")
        except Exception as e:
            print(f"[MinIO] Warning: Backup échoué ({e}), mais on continue vers W&B.")

        # --- B. UPLOAD W&B (La partie critique pour le pipeline) ---
        artifact = wandb.Artifact(
            name=artifact_name,
            type='model',
            metadata=metadata
        )

        # CHANGEMENT ICI : On utilise add_file au lieu de add_reference
        # Cela envoie le fichier vers le cloud W&B
        artifact.add_file(local_path)

        run.log_artifact(artifact, aliases=aliases or [])
        print(f"[W&B] Modèle uploadé et versionné : {artifact_name}")

    def download_artifact(self, artifact_name, local_path, run):
        """
        Télécharge le fichier directement depuis W&B.
        Si artifact_name contient déjà une version (ex: :v0, :latest), l'utilise directement.
        Sinon, ajoute :latest par défaut.
        """
        # Vérifier si une version est déjà spécifiée
        if ":" in artifact_name:
            full_artifact_name = artifact_name
        else:
            full_artifact_name = artifact_name + ":latest"
            
        print(f"[W&B] Téléchargement de {full_artifact_name} ...")

        # 1. On récupère l'artefact
        artifact = run.use_artifact(full_artifact_name)

        # 2. On laisse W&B télécharger le fichier dans un dossier cache
        # root=os.path.dirname(local_path) force le download dans le dossier voulu
        download_dir = artifact.download(root=os.path.dirname(local_path))

        # 3. W&B garde le nom de fichier original lors de l'upload.
        # On doit renommer ou s'assurer que le fichier a le nom attendu par notre script.

        # On cherche le fichier .pth dans le dossier de téléchargement
        downloaded_files = [f for f in os.listdir(download_dir) if f.endswith('.pth')]

        if not downloaded_files:
            raise FileNotFoundError("Aucun fichier .pth trouvé dans l'artefact W&B")

        # Le fichier téléchargé par W&B
        source_file = os.path.join(download_dir, downloaded_files[0])

        # Si le nom n'est pas celui attendu par local_path, on renomme
        if os.path.abspath(source_file) != os.path.abspath(local_path):
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(source_file, local_path)

        return local_path