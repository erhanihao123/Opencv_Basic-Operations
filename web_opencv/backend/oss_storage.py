import os
import logging
import oss2
from datetime import datetime

logger = logging.getLogger(__name__)


class OSSStorage:
    def __init__(self):
        self.access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        self.access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
        self.endpoint = os.getenv('OSS_ENDPOINT')
        self.bucket_name = os.getenv('OSS_BUCKET_NAME')
        self.enabled = all([self.access_key_id, self.access_key_secret, self.endpoint, self.bucket_name])
        
        if self.enabled:
            try:
                auth = oss2.Auth(self.access_key_id, self.access_key_secret)
                self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
                logger.info(f"OSS storage initialized: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OSS storage: {e}")
                self.enabled = False
        else:
            logger.info("OSS storage not configured, skipping initialization")

    def is_enabled(self):
        return self.enabled

    def upload_image(self, image_bytes, filename=None, prefix='images/'):
        if not self.enabled:
            logger.warning("OSS storage is not enabled")
            return None
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}.png"
            
            oss_key = f"{prefix}{filename}"
            self.bucket.put_object(oss_key, image_bytes)
            
            url = f"https://{self.bucket_name}.{self.endpoint}/{oss_key}"
            logger.info(f"Image uploaded to OSS: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload image to OSS: {e}")
            return None

    def download_image(self, oss_key):
        if not self.enabled:
            logger.warning("OSS storage is not enabled")
            return None
        
        try:
            result = self.bucket.get_object(oss_key)
            return result.read()
        except Exception as e:
            logger.error(f"Failed to download image from OSS: {e}")
            return None

    def list_images(self, prefix='images/'):
        if not self.enabled:
            logger.warning("OSS storage is not enabled")
            return []
        
        try:
            images = []
            for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
                url = f"https://{self.bucket_name}.{self.endpoint}/{obj.key}"
                images.append({
                    'key': obj.key,
                    'url': url,
                    'size': obj.size,
                    'last_modified': obj.last_modified
                })
            return images
        except Exception as e:
            logger.error(f"Failed to list images from OSS: {e}")
            return []

    def delete_image(self, oss_key):
        if not self.enabled:
            logger.warning("OSS storage is not enabled")
            return False
        
        try:
            self.bucket.delete_object(oss_key)
            logger.info(f"Image deleted from OSS: {oss_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image from OSS: {e}")
            return False

    def upload_model(self, model_bytes, filename):
        return self.upload_image(model_bytes, filename, prefix='models/')

    def download_model(self, filename):
        return self.download_image(f'models/{filename}')