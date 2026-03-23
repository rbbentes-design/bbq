"""
Downloader de imagens para storage local.

Baixa URLs de imagens e salva em disco, retornando mapa URL -> Path local.
Usa httpx com timeout curto; falhas sao silenciosas (imagem fica como URL).
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

import httpx

from app.audit.logger import get_logger
from app.utils.hashing import sha256_of_str

_log = get_logger("utils.image_downloader")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def _ext_from(url: str, content_type: str) -> str:
    """Devolve extensao (.png, .jpg, .gif, .webp ...) a partir da URL ou content-type."""
    # Tenta pela URL primeiro
    path = url.split("?")[0].split("#")[0]
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif"}:
        return suffix
    # Fallback: content-type
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".jpg"
    # mimetypes usa .jpeg; normaliza para .jpg
    return ".jpg" if ext == ".jpeg" else ext


def download_images(urls: list[str], dest_dir: Path) -> dict[str, Path]:
    """
    Baixa uma lista de URLs para dest_dir.

    Args:
        urls: Lista de URLs de imagens (duplicatas sao ignoradas).
        dest_dir: Diretorio de destino (criado se nao existir).

    Returns:
        Dicionario {url: path_local} apenas das imagens baixadas com sucesso.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    seen: set[str] = set()

    with httpx.Client(
        timeout=15,
        follow_redirects=True,
        headers=_HEADERS,
    ) as client:
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)

            # Arquivo ja existe no disco?
            file_hash = sha256_of_str(url)[:20]
            # Descobre extensao tentando HEAD primeiro (rapido)
            existing = list(dest_dir.glob(f"{file_hash}.*"))
            if existing:
                result[url] = existing[0]
                continue

            try:
                resp = client.get(url, headers={"Referer": url})
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "image/jpeg")
                ext = _ext_from(url, ct)
                local_path = dest_dir / f"{file_hash}{ext}"
                local_path.write_bytes(resp.content)
                result[url] = local_path
                _log.debug("image_downloaded", url=url, path=str(local_path))
            except Exception as exc:
                _log.warning("image_download_failed", url=url, error=str(exc))

    _log.info("images_done", total=len(urls), downloaded=len(result))
    return result
