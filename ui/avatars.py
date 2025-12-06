"""Avatar helper functions."""
from pathlib import Path


def get_logo_avatar():
    """Get the logo path for avatar, trying multiple locations."""
    logo_paths = ["assets/logo.png", "assets/logo.jpg", "logo.png", "logo.jpg"]
    for logo_path in logo_paths:
        if Path(logo_path).exists():
            try:
                return logo_path
            except Exception:
                continue
    return None
