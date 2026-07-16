from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any


HF_BASE_URL = "https://huggingface.co"
HF_DOWNLOAD_BASE_URL = os.environ.get("HF_DOWNLOAD_BASE_URL", HF_BASE_URL).rstrip("/")


@dataclass(frozen=True)
class Artifact:
    model: str
    variant: str
    repo_id: str
    filename: str
    relative_path: str
    source: str = "official"
    required: bool = True
    blocked_reason: str | None = None

    @property
    def url(self) -> str:
        quoted = "/".join(urllib.parse.quote(part) for part in self.filename.split("/"))
        return f"{HF_DOWNLOAD_BASE_URL}/{self.repo_id}/resolve/main/{quoted}"


ARTIFACTS = (
    Artifact("olmoearth", "nano", "allenai/OlmoEarth-v1-Nano", "config.json", "olmoearth/nano/config.json"),
    Artifact("olmoearth", "nano", "allenai/OlmoEarth-v1-Nano", "weights.pth", "olmoearth/nano/weights.pth"),
    Artifact("olmoearth", "tiny", "allenai/OlmoEarth-v1-Tiny", "config.json", "olmoearth/tiny/config.json"),
    Artifact("olmoearth", "tiny", "allenai/OlmoEarth-v1-Tiny", "weights.pth", "olmoearth/tiny/weights.pth"),
    Artifact("olmoearth", "base", "allenai/OlmoEarth-v1-Base", "config.json", "olmoearth/base/config.json"),
    Artifact("olmoearth", "base", "allenai/OlmoEarth-v1-Base", "weights.pth", "olmoearth/base/weights.pth"),
    Artifact("olmoearth", "large", "allenai/OlmoEarth-v1-Large", "config.json", "olmoearth/large/config.json"),
    Artifact("olmoearth", "large", "allenai/OlmoEarth-v1-Large", "weights.pth", "olmoearth/large/weights.pth"),
    Artifact("anysat", "base", "g-astruc/AnySat", "models/AnySat.pth", "anysat/base/AnySat.pth"),
    Artifact("clay", "large-v1.5", "made-with-clay/Clay", "v1.5/clay-v1.5.ckpt", "clay/large-v1.5/clay-v1.5.ckpt"),
    Artifact("copernicusfm", "base", "wangyi111/Copernicus-FM", "CopernicusFM_ViT_base_varlang_e100.pth", "copernicusfm/base/CopernicusFM_ViT_base_varlang_e100.pth"),
    Artifact("copernicusfm", "large", "wangyi111/Copernicus-FM", "CopernicusFM_ViT_large_varlang_e100.pth", "copernicusfm/large/CopernicusFM_ViT_large_varlang_e100.pth"),
    Artifact("croma", "base", "antofuller/CROMA", "CROMA_base.pt", "croma/base/CROMA_base.pt"),
    Artifact("croma", "large", "antofuller/CROMA", "CROMA_large.pt", "croma/large/CROMA_large.pt"),
    Artifact("galileo", "nano", "nasaharvest/galileo", "models/nano/config.json", "galileo/nano/config.json"),
    Artifact("galileo", "nano", "nasaharvest/galileo", "models/nano/encoder.pt", "galileo/nano/encoder.pt"),
    Artifact("galileo", "tiny", "nasaharvest/galileo", "models/tiny/config.json", "galileo/tiny/config.json"),
    Artifact("galileo", "tiny", "nasaharvest/galileo", "models/tiny/encoder.pt", "galileo/tiny/encoder.pt"),
    Artifact("galileo", "base", "nasaharvest/galileo", "models/base/config.json", "galileo/base/config.json"),
    Artifact("galileo", "base", "nasaharvest/galileo", "models/base/encoder.pt", "galileo/base/encoder.pt"),
    Artifact("panopticon", "vitb14", "lewaldm/panopticon", "panopticon_vitb14_teacher.pth", "panopticon/vitb14/panopticon_vitb14_teacher.pth"),
    Artifact("presto", "default", "nasaharvest/presto", "default_model.pt", "presto/default/default_model.pt"),
    Artifact("prithviv2", "300m", "ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "config.json", "prithviv2/300m/config.json"),
    Artifact("prithviv2", "300m", "ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "Prithvi_EO_V2_300M.pt", "prithviv2/300m/Prithvi_EO_V2_300M.pt"),
    Artifact("prithviv2", "600m", "ibm-nasa-geospatial/Prithvi-EO-2.0-600M", "config.json", "prithviv2/600m/config.json"),
    Artifact("prithviv2", "600m", "ibm-nasa-geospatial/Prithvi-EO-2.0-600M", "Prithvi_EO_V2_600M.pt", "prithviv2/600m/Prithvi_EO_V2_600M.pt"),
    Artifact("satlas", "base-s2", "allenai/satlas-pretrain", "sentinel2_swinb_si_ms.pth", "satlas/base/sentinel2_swinb_si_ms.pth"),
    Artifact("satlas", "base-s1", "allenai/satlas-pretrain", "sentinel1_swinb_si.pth", "satlas/base/sentinel1_swinb_si.pth"),
    Artifact("satlas", "base-landsat", "allenai/satlas-pretrain", "landsat_swinb_si.pth", "satlas/base/landsat_swinb_si.pth"),
    Artifact("terramind", "base", "ibm-esa-geospatial/TerraMind-1.0-base", "TerraMind_v1_base.pt", "terramind/base/TerraMind_v1_base.pt"),
    Artifact("terramind", "large", "ibm-esa-geospatial/TerraMind-1.0-large", "TerraMind_v1_large.pt", "terramind/large/TerraMind_v1_large.pt"),
    Artifact("tessera", "v1", "isaaccorley/tessera", "best_model_fsdp_20250427_084307.pt", "tessera/v1/best_model_fsdp_20250427_084307.pt", source="community mirror of access-restricted official checkpoint"),
    Artifact("dinov3", "vitb16-web", "facebook/dinov3-vitb16-pretrain-lvd1689m", "model.safetensors", "dinov3/vitb16-web/model.safetensors", blocked_reason="DINOv3 license acceptance and Hugging Face authentication required"),
    Artifact("dinov3", "vitl16-web", "facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors", "dinov3/vitl16-web/model.safetensors", blocked_reason="DINOv3 license acceptance and Hugging Face authentication required"),
    Artifact("dinov3", "vith16plus-web", "facebook/dinov3-vith16plus-pretrain-lvd1689m", "model.safetensors", "dinov3/vith16plus-web/model.safetensors", blocked_reason="DINOv3 license acceptance and Hugging Face authentication required"),
    Artifact("dinov3", "vit7b16-web", "facebook/dinov3-vit7b16-pretrain-lvd1689m", "model.safetensors.index.json", "dinov3/vit7b16-web/model.safetensors.index.json", blocked_reason="DINOv3 license acceptance, authentication, and about 27 GB required"),
    Artifact("dinov3", "vitl16-sat", "facebook/dinov3-vitl16-pretrain-sat493m", "model.safetensors", "dinov3/vitl16-sat/model.safetensors", blocked_reason="DINOv3 license acceptance and Hugging Face authentication required"),
    Artifact("dinov3", "vit7b16-sat", "facebook/dinov3-vit7b16-pretrain-sat493m", "model.safetensors.index.json", "dinov3/vit7b16-sat/model.safetensors.index.json", blocked_reason="DINOv3 license acceptance, authentication, and about 27 GB required"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GeoFM checkpoints with resume and verification.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="*", help="Optional model-family filter.")
    parser.add_argument("--include-blocked", action="store_true", help="Try gated artifacts using HF_TOKEN.")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent file downloads.")
    parser.add_argument("--list", action="store_true", help="Only write metadata and print the plan.")
    return parser.parse_args()


def hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def request_json(url: str, *, data: dict[str, Any] | None = None) -> Any:
    payload = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", **hf_headers()})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def artifact_metadata(artifact: Artifact) -> dict[str, Any]:
    repo = urllib.parse.quote(artifact.repo_id, safe="/")
    url = f"{HF_BASE_URL}/api/models/{repo}/paths-info/main"
    result = request_json(url, data={"paths": [artifact.filename], "expand": True})
    if not result:
        raise RuntimeError(f"Artifact not found: {artifact.repo_id}/{artifact.filename}")
    item = result[0]
    lfs = item.get("lfs") or {}
    return {
        "expected_size": item.get("size"),
        "sha256": lfs.get("oid"),
        "git_oid": item.get("oid"),
    }


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(artifact: Artifact, destination: Path, log_path: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = hf_headers()
    aria2 = shutil.which("aria2c")
    if aria2:
        command = [
            aria2,
            "--continue=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=16M",
            "--max-tries=10",
            "--retry-wait=5",
            "--timeout=60",
            "--summary-interval=5",
            "--console-log-level=warn",
            f"--dir={destination.parent}",
            f"--out={destination.name}",
        ]
        if "Authorization" in headers:
            command.append(f"--header=Authorization: {headers['Authorization']}")
        command.append(artifact.url)
    else:
        curl = shutil.which("curl") or shutil.which("curl.exe")
        if curl is None:
            raise RuntimeError("Neither aria2c nor curl is available.")
        command = [curl, "--location", "--fail", "--retry", "10", "--retry-delay", "5", "--continue-at", "-", "--progress-bar"]
        if "Authorization" in headers:
            command.extend(["--header", f"Authorization: {headers['Authorization']}"])
        command.extend(["--output", str(destination), artifact.url])
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command[:3])} ... {artifact.url}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Downloader exited with code {completed.returncode}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "download_status.json"
    log_path = root / "download.log"
    selected = [item for item in ARTIFACTS if not args.models or item.model in set(args.models)]
    statuses: list[dict[str, Any]] = []
    total_size = 0

    print("Resolving model metadata...", flush=True)
    for artifact in selected:
        row = {**asdict(artifact), "url": artifact.url, "status": "pending"}
        if artifact.blocked_reason and not args.include_blocked:
            row["status"] = "blocked"
            statuses.append(row)
            continue
        try:
            row.update(artifact_metadata(artifact))
            total_size += row.get("expected_size") or 0
        except (RuntimeError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            row["status"] = "metadata_error"
            row["error"] = str(exc)
        statuses.append(row)
    write_json(status_path, statuses)
    downloadable = [row for row in statuses if row["status"] == "pending"]
    print(f"Plan: {len(downloadable)} files, {human_size(total_size)}; {len(statuses) - len(downloadable)} blocked/unavailable.", flush=True)
    print(f"Status: {status_path}", flush=True)
    if args.list:
        return 0

    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    status_lock = Lock()
    artifacts_by_path = {item.relative_path: item for item in selected}

    def save_status() -> None:
        with status_lock:
            write_json(status_path, statuses)

    def process_artifact(index: int, row: dict[str, Any]) -> None:
        artifact = artifacts_by_path[row["relative_path"]]
        destination = root / artifact.relative_path
        expected_size = row.get("expected_size")
        row["local_path"] = str(destination)
        existing_size = destination.stat().st_size if destination.exists() else 0
        if destination.exists() and expected_size == existing_size:
            if args.verify_existing and row.get("sha256"):
                row["status"] = "verifying"
                save_status()
                actual = sha256_file(destination)
                row["actual_sha256"] = actual
                row["status"] = "complete" if actual == row["sha256"] else "checksum_error"
            else:
                row["status"] = "complete"
            save_status()
            print(f"[{index}/{len(downloadable)}] Already complete: {artifact.model}/{artifact.variant}", flush=True)
            return

        row["status"] = "downloading"
        row["downloaded_bytes"] = existing_size
        row["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_status()
        print(f"[{index}/{len(downloadable)}] Downloading {artifact.model}/{artifact.variant}: {human_size(expected_size)}", flush=True)
        try:
            download(artifact, destination, log_path)
            actual_size = destination.stat().st_size
            row["downloaded_bytes"] = actual_size
            if expected_size is not None and actual_size != expected_size:
                raise RuntimeError(f"Size mismatch: expected {expected_size}, got {actual_size}")
            row["status"] = "verifying"
            save_status()
            if row.get("sha256"):
                actual_hash = sha256_file(destination)
                row["actual_sha256"] = actual_hash
                if actual_hash != row["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch: expected {row['sha256']}, got {actual_hash}")
            row["status"] = "complete"
            row["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            save_status()
            print(f"FAILED: {artifact.model}/{artifact.variant}: {exc}", file=sys.stderr, flush=True)
            raise
        save_status()

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_artifact, index, row): row
            for index, row in enumerate(downloadable, start=1)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future]["relative_path"], str(exc)))

    complete = sum(row["status"] == "complete" for row in statuses)
    blocked = sum(row["status"] == "blocked" for row in statuses)
    print(f"Finished: {complete} complete, {blocked} blocked, {len(failures)} failed.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
