from __future__ import annotations

import json
import urllib.request
import urllib.error
import ssl
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

PRESETS_DIR = Path("presets")


def save_preset(name: str, params: dict) -> Path:
    PRESETS_DIR.mkdir(exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "preset"
    path = PRESETS_DIR / f"{safe}.json"
    p = dict(params)
    if isinstance(p.get("color"), tuple):
        p["color"] = list(p["color"])
    with open(path, "w") as f:
        json.dump(p, f, indent=2)
    return path


def load_preset(name: str) -> Optional[dict]:
    n = name if name.endswith(".json") else f"{name}.json"
    path = PRESETS_DIR / n
    if not path.exists():
        return None
    try:
        with open(path) as f:
            p = json.load(f)
        if "color" in p and isinstance(p["color"], list):
            p["color"] = tuple(p["color"])
        return p
    except Exception:
        return None


def list_presets() -> list[str]:
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def delete_preset(name: str) -> bool:
    n = name if name.endswith(".json") else f"{name}.json"
    path = PRESETS_DIR / n
    if path.exists():
        path.unlink()
        return True
    return False


class DownloadPresetsWorker(QThread):
    progress = Signal(str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, repo: str = "shahil-sk/dither-guy", branch: str = "master"):
        super().__init__()
        self.repo = repo
        self.branch = branch

    def run(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        api_url = f"https://api.github.com/repos/{self.repo}/contents/presets?ref={self.branch}"
        try:
            self.progress.emit("Fetching preset list from GitHub...")
            req = urllib.request.Request(api_url, headers={'User-Agent': 'DitherGuy-App'})
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                data = json.loads(response.read().decode())
            
            if not isinstance(data, list):
                self.error.emit("Presets folder not found on GitHub.")
                return

            downloaded = []
            PRESETS_DIR.mkdir(exist_ok=True)
            
            files_to_download = [item for item in data if item.get("type") == "file" and item.get("name", "").endswith(".json")]
            
            if not files_to_download:
                self.error.emit("No preset files found in the repository.")
                return

            for i, item in enumerate(files_to_download):
                name = item["name"]
                file_url = item.get("download_url")
                if not file_url:
                    continue
                    
                self.progress.emit(f"Downloading {name}... ({i+1}/{len(files_to_download)})")
                freq = urllib.request.Request(file_url, headers={'User-Agent': 'DitherGuy-App'})
                with urllib.request.urlopen(freq, timeout=10, context=ctx) as fres:
                    content = fres.read().decode()
                    save_path = PRESETS_DIR / name
                    with open(save_path, "w") as f:
                        f.write(content)
                    downloaded.append(name)
                    
            self.finished.emit(downloaded)
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.error.emit("Presets folder not found on GitHub repository.")
            else:
                self.error.emit(f"HTTP Error: {e.code} {e.reason}")
        except Exception as e:
            self.error.emit(f"Failed to download presets: {str(e)}")
