"""
Utility functions for PaperBanana framework.
"""
import base64
import mimetypes
import os
from typing import Optional


def save_binary_file(file_name: str, data: bytes) -> str:
    """
    Save binary data to a file.
    
    Args:
        file_name: Name of the file to save
        data: Binary data to write
        
    Returns:
        Path to the saved file
    """
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")
    return file_name


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_mime_type(file_path: str) -> Optional[str]:
    """
    Get MIME type for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string or None
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type
