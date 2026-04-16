from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status

from app.config import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

 

def get_runtime_registry_root() -> Path:
    return safe_mkdir(Path(settings.registry_root))


def get_dvc_registry_root() -> Path:
    return safe_mkdir(Path(settings.dvc_registry_root))


def get_uploads_root() -> Path:
    return safe_mkdir(get_runtime_registry_root() / settings.uploads_dir_name)


def get_production_root() -> Path:
    return safe_mkdir(get_runtime_registry_root() / settings.production_dir_name)


def get_archive_root() -> Path:
    return safe_mkdir(get_runtime_registry_root() / settings.archive_dir_name)


def get_dvc_uploads_root() -> Path:
    return safe_mkdir(get_dvc_registry_root() / settings.uploads_dir_name)


def get_dvc_production_root() -> Path:
    return safe_mkdir(get_dvc_registry_root() / settings.production_dir_name)


def get_dvc_archive_root() -> Path:
    return safe_mkdir(get_dvc_registry_root() / settings.archive_dir_name)


def current_model_json_path() -> Path:
    return get_runtime_registry_root() / settings.production_dir_name / "current_model.json"


def dvc_current_model_json_path() -> Path:
    return get_dvc_production_root() / "current_model.json"


def candidate_dir(candidate_id: str) -> Path:
    return get_uploads_root() / candidate_id


def dvc_candidate_dir(candidate_id: str) -> Path:
    return get_dvc_uploads_root() / candidate_id


 

def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required JSON file: {path.name}",
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def extract_metric(results: dict[str, Any]) -> tuple[str, float]:
    primary = settings.comparison_primary_metric
    fallback = settings.comparison_fallback_metric

    if primary in results and isinstance(results[primary], (int, float)):
        return primary, float(results[primary])

    if fallback in results and isinstance(results[fallback], (int, float)):
        return fallback, float(results[fallback])

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"results.json must contain '{primary}' "
            f"or '{fallback}' as a numeric field."
        ),
    )


def save_upload_file(destination: Path, upload_file: UploadFile) -> None:
    safe_mkdir(destination.parent)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def validate_token(auth_header: str | None) -> None:
    expected = f"Bearer {settings.registry_api_token}"
    if auth_header != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )



def clear_directory_contents(path: Path) -> None:
    safe_mkdir(path)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)


def copy_directory_contents(src: Path, dst: Path) -> None:
    safe_mkdir(dst)
    clear_directory_contents(dst)

    if not src.exists():
        return

    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        safe_mkdir(dst.parent)
        shutil.copy2(src, dst)


def sync_candidate_to_runtime(candidate_id: str) -> None:
    copy_directory_contents(dvc_candidate_dir(candidate_id), candidate_dir(candidate_id))


def sync_production_to_runtime() -> None:
    copy_directory_contents(get_dvc_production_root(), get_production_root())


def sync_archive_to_runtime() -> None:
    copy_directory_contents(get_dvc_archive_root(), get_archive_root())


def sync_all_dvc_to_runtime() -> None:
    sync_production_to_runtime()
    sync_archive_to_runtime()

    dvc_uploads = get_dvc_uploads_root()
    runtime_uploads = get_uploads_root()
    safe_mkdir(runtime_uploads)

    for child in dvc_uploads.iterdir():
        if child.is_dir():
            copy_directory_contents(child, runtime_uploads / child.name)


 

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd or repo_root()),
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Command failed: {' '.join(cmd)}\n"
                f"stdout:\n{e.stdout}\n"
                f"stderr:\n{e.stderr}"
            ),
        ) from e


def dvc_track_and_push_registry() -> None:
    if not settings.dvc_enabled:
        return

    root = repo_root()
    relative_registry_path = "storage/registry"

    run_cmd(["dvc", "add", relative_registry_path], cwd=root)
    run_cmd(["git", "add", "storage/registry.dvc", "storage/.gitignore"], cwd=root)

    if settings.dvc_auto_push:
        run_cmd(["dvc", "push"], cwd=root)


 

def save_candidate_files(
    candidate_id: str,
    run_name: str | None,
    results_file: UploadFile,
    best_model_file: UploadFile,
) -> tuple[Path, dict[str, Any], str, float]:
    dest_dir = safe_mkdir(dvc_candidate_dir(candidate_id))

    save_upload_file(dest_dir / "results.json", results_file)
    save_upload_file(dest_dir / "best_model.pth", best_model_file)

    results = read_json_file(dest_dir / "results.json")
    metric_name, metric_value = extract_metric(results)

    metadata = {
        "candidate_id": candidate_id,
        "run_name": run_name,
        "uploaded_at": utc_now_iso(),
        "metric_name": metric_name,
        "metric_value": metric_value,
    }
    write_json_file(dest_dir / "metadata.json", metadata)

    dvc_track_and_push_registry()
    sync_candidate_to_runtime(candidate_id)

    return dest_dir, results, metric_name, metric_value


 

def load_current_production_metadata() -> dict[str, Any] | None:
    path = dvc_current_model_json_path()
    if not path.exists():
        return None
    return read_json_file(path)


def archive_existing_production() -> Path | None:
    production_root = get_dvc_production_root()
    current_json = dvc_current_model_json_path()

    if not current_json.exists():
        return None

    production_files = [p for p in production_root.iterdir() if p.name != "current_model.json"]
    if not production_files:
        return None

    current_meta = read_json_file(current_json)
    archive_name = current_meta.get("candidate_id") or f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dest = safe_mkdir(get_dvc_archive_root() / archive_name)

    for item in production_files:
        target = archive_dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    shutil.copy2(current_json, archive_dest / "current_model.json")

    for item in production_files:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

    current_json.unlink(missing_ok=True)
    return archive_dest



def promote_candidate(candidate_id: str) -> dict[str, Any]:
    source_dir = dvc_candidate_dir(candidate_id)
    if not source_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate '{candidate_id}' not found in uploads.",
        )

    results = read_json_file(source_dir / "results.json")
    candidate_metric_name, candidate_metric_value = extract_metric(results)

    existing_prod = load_current_production_metadata()
    archive_dest = None

    if existing_prod is not None:
        prod_metric_name = existing_prod.get("metric_name")
        prod_metric_value = existing_prod.get("metric_value")

        if prod_metric_name != candidate_metric_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Metric mismatch between candidate and production. "
                    f"candidate uses '{candidate_metric_name}', "
                    f"production uses '{prod_metric_name}'."
                ),
            )

        if not isinstance(prod_metric_value, (int, float)):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Production metadata is invalid: metric_value missing or non-numeric.",
            )

        if float(candidate_metric_value) <= float(prod_metric_value):
            return {
                "promoted": False,
                "reason": "Candidate is not better than current production.",
                "candidate_metric_name": candidate_metric_name,
                "candidate_metric_value": candidate_metric_value,
                "production_metric_name": prod_metric_name,
                "production_metric_value": float(prod_metric_value),
                "archive_dir": None,
                "current_model": existing_prod,
            }

        archive_dest = archive_existing_production()

    production_root = get_dvc_production_root()
    clear_directory_contents(production_root)

    for filename in ["best_model.pth", "results.json"]:
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, production_root / filename)

    metadata = read_json_file(source_dir / "metadata.json")
    current_model = {
        "candidate_id": candidate_id,
        "run_name": metadata.get("run_name"),
        "promoted_at": utc_now_iso(),
        "metric_name": candidate_metric_name,
        "metric_value": candidate_metric_value,
        "source_upload_dir": str(source_dir),
        "production_model_path": str(production_root / "best_model.pth"),
        "results_path": str(production_root / "results.json"),
    }
    write_json_file(dvc_current_model_json_path(), current_model)

    dvc_track_and_push_registry()
    sync_production_to_runtime()
    sync_archive_to_runtime()

    return {
        "promoted": True,
        "reason": "Candidate promoted to production.",
        "candidate_metric_name": candidate_metric_name,
        "candidate_metric_value": candidate_metric_value,
        "production_metric_name": candidate_metric_name,
        "production_metric_value": candidate_metric_value,
        "archive_dir": str(archive_dest) if archive_dest else None,
        "current_model": current_model,
    }