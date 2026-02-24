import base64
import hashlib
from typing import Optional

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class Encryption:
    """Encryption utilities for sensitive data."""
    
    def __init__(self, key: Optional[str] = None):
        self._key = key
        self._fernet = None
        
        if CRYPTO_AVAILABLE and key:
            self._init_fernet(key)
    
    def _init_fernet(self, key: str) -> None:
        """Initialize Fernet with key."""
        try:
            key_bytes = key.encode() if len(key) < 44 else base64.urlsafe_b64decode(key)
            self._fernet = Fernet(key_bytes)
        except Exception:
            derived = self._derive_key(key)
            self._fernet = Fernet(derived)
    
    def _derive_key(self, password: str, salt: bytes = b"omnimind_salt") -> bytes:
        """Derive key from password using PBKDF2."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")
        return Fernet.generate_key().decode()
    
    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")
        if not self._fernet:
            raise ValueError("Encryption not initialized with key")
        
        encrypted = self._fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")
        if not self._fernet:
            raise ValueError("Encryption not initialized with key")
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception:
            raise ValueError("Decryption failed")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password) == hashed
    
    @staticmethod
    def is_available() -> bool:
        return CRYPTO_AVAILABLE


encryption = Encryption()
