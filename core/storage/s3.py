import boto3
import uuid
import logging
from core.config.settings import settings

logger = logging.getLogger(__name__)

class S3Folders:
    AVATARS = settings.S3_AVATAR_FOLDER
    WARDROBE = settings.S3_WARDROBE_FOLDER
    OUTFITS = "outfits"
    TRYON = "tryons"
    TEMP = "temp"

class S3Storage:
    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.AWS_BUCKET_NAME
        self.region = settings.AWS_REGION

    def upload(
        self,
        image_bytes: bytes,
        folder: str,
        extension: str = "png",
        content_type: str = "image/png"
    ) -> dict:
        """Upload image to S3 — returns key and public URL"""
        try:
            file_key = f"{folder}/{uuid.uuid4()}.{extension}"

            self.client.put_object(
                Bucket=self.bucket,
                Key=file_key,
                Body=image_bytes,
                ContentType=content_type
            )

            url = self.get_url(file_key)
            return { "key": file_key, "url": url }

        except Exception as e:
            logger.exception(f"S3 upload failed: {e}")
            raise

    def delete(self, file_key: str) -> bool:
        """Delete file from S3 by key"""
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=file_key
            )
            return True
        except Exception as e:
            logger.exception(f"S3 delete failed: {e}")
            return False

    def get_url(self, file_key: str) -> str:
        """Generate public URL from file key"""
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{file_key}"

storage = S3Storage()
