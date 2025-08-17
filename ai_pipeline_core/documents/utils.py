import re
from urllib.parse import urlparse


def sanitize_url(url: str) -> str:
    """
    Sanitize URL or query string for use in filenames.
    Removes or replaces characters that are invalid in filenames.
    """
    # Remove protocol if it's a URL
    if url.startswith(("http://", "https://")):
        parsed = urlparse(url)
        # Use domain + path
        url = parsed.netloc + parsed.path

    # Replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", url)

    # Replace multiple underscores with single one
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    # Limit length to prevent too long filenames
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    # Ensure we have something
    if not sanitized:
        sanitized = "unnamed"

    return sanitized
