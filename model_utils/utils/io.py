import os
import json
import socket
from pathlib import Path

def get_weights_path(model_type: str, encoder_name: str) -> str:
    """
    Retrieve the path to the weights file for a given model name.
    This function looks up the path to the weights file in a local checkpoint
    registry (local_ckpts.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory.
    
    Parameters
    ----------
    model_type : str
        The type of model ('patch', 'slide', or 'seg').
    encoder_name : str
        The name of the model whose weights path is to be retrieved.
        
    Returns
    -------
    str
        The absolute path to the weights file.
    """

    root = Path(__file__).parent.parent

    registry_path = os.path.join(root, "model_weights.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)

    path = registry.get(encoder_name)    
    return path



def has_internet_connection(timeout: float = 3.0) -> bool:
    endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
    
    if endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        endpoint = urlparse(endpoint).netloc
    
    try:
        # Fast socket-level check
        socket.create_connection((endpoint, 443), timeout=timeout)
        return True
    except OSError:
        pass

    try:
        # Fallback HTTP-level check (if requests is available)
        import requests
        url = f"https://{endpoint}" if not endpoint.startswith(("http://", "https://")) else endpoint
        r = requests.head(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False