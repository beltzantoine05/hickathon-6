import os
import boto3
import wandb
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Charge les variables depuis le fichier .env à la racine
load_dotenv()

class ArtifactManager:
    """
    Gère le cycle de vie des modèles : 
    1. Stockage physique sur MinIO (Souveraineté des données)
    2. Versioning et Lineage sur W&B (Tracking)
    """
    def __init__(self, bucket_name=os.getenv("MINIO_BUCKET")):
        self.bucket = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_ENDPOINT'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
            aws_session_token=os.getenv('MINIO_SESSION_TOKEN', None)
<<<<<<< HEAD
=======

>>>>>>> d635570b24e990906c9dbd89a82e2aa27306e8fa
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Vérifie si le bucket existe, sinon le crée."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
                print(f"[MinIO] Bucket '{self.bucket}' created.")
            except Exception as e:
                print(f"[MinIO] Error creating bucket: {e}")

    def log_model(self, local_path, artifact_name, run, metadata=None, aliases=None):
        """
        Upload le modèle sur MinIO et enregistre la référence dans W&B.
        
        Args:
            local_path (str): Chemin local du fichier .pth
            artifact_name (str): Nom de l'artefact (ex: 'dae-encoder-fold-0')
            run (wandb.Run): L'objet run W&B actif
            metadata (dict): Métadonnées (loss, epoch, config...)
            aliases (list): Tags pour W&B (ex: ['latest', 'best'])
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model file not found: {local_path}")

        # 1. Construction du chemin S3 unique (namespace par Run ID pour éviter les écrasements)
        # Structure: models/<run_id>/<artifact_name>.pth
        s3_key = f"models/{run.id}/{artifact_name}.pth"
        
        # 2. Upload sur MinIO
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
            s3_uri = f"s3://{self.bucket}/{s3_key}"
        except Exception as e:
            print(f"[Error] Failed to upload to MinIO: {e}")
            return

        # 3. Création de l'artefact W&B (Reference Artifact)
        # On ne stocke pas le binaire sur W&B, juste le pointeur vers MinIO
        artifact = wandb.Artifact(
            name=artifact_name,
            type='model',
            metadata=metadata
        )
        artifact.add_reference(s3_uri)

        # 4. Log dans W&B
        run.log_artifact(artifact, aliases=aliases or [])
        print(f"[Artifact] Model saved to {s3_uri} and tracked in W&B as '{artifact_name}'")

    def download_artifact(self, artifact_name, local_path, run):
        """
        Récupère un artefact.
        Utilise W&B pour résoudre la version 'latest' et récupérer l'URI S3,
        puis utilise boto3 pour le téléchargement physique.
        """
        # 1. Signale à W&B qu'on utilise cet artefact (crée le lien dans le graph)
        artifact = run.use_artifact(artifact_name + ":latest")
        
        # 2. Récupère l'URI S3 depuis le manifeste de l'artefact
        # Les entrées sont dans un dictionnaire, on prend la première (notre fichier .pth)
        entry = list(artifact.manifest.entries.values())[0]
        ref_url = entry.ref  # ex: s3://bucket-name/models/run_id/model.pth
        
        # 3. Parsing de l'URI S3
        # Format s3://bucket/key
        parts = ref_url.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        
        # 4. Téléchargement via notre client Boto3 configuré
        print(f"[MinIO] Downloading from {ref_url} to {local_path}")
        self.s3_client.download_file(bucket, key, local_path)
        
        return local_path