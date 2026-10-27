"""Frozen CPU sentence embeddings for observed names/addresses only.

The real encoder is imported lazily. Unit tests can exercise cache/feature logic
without importing Sentence Transformers or contacting any model service.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import pair_features as pf
import prepare_candidates as pc


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_FIELDS = ("company_name_normalized", "address_normalized")
EMBEDDING_NAMES = ("company_name_embedding_cosine", "address_embedding_cosine")
FEATURE_NAMES = (*pf.FEATURE_NAMES, *EMBEDDING_NAMES)
BATCH_SIZE = 32
CPU_THREADS = 2


def text_value(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("Embedding input must be normalised observed text")
    return value.strip()


def preprocessing_version() -> str:
    source = "\n".join(inspect.getsource(function) for function in (
        pc.normalize_text, pc.normalize_company_name, pc.normalize_address
    ))
    return "stage2-normalized-v1:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheContext:
    model_revision: str
    preprocessing_version: str
    runtime_version: str
    model_id: str = MODEL_ID
    dimension: int = 384
    normalize_embeddings: bool = True


class EmbeddingCache:
    def __init__(self, directory: Path, context: CacheContext):
        self.directory = directory
        self.context = context

    def identity(self, text: str) -> dict:
        return {**asdict(self.context), "text": text}

    def path_for(self, text: str) -> Path:
        encoded = json.dumps(self.identity(text), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return self.directory / (hashlib.sha256(encoded).hexdigest() + ".npz")

    def validate_vector(self, vector) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        pc.require(vector.shape == (self.context.dimension,) and np.isfinite(vector).all(),
                   "Invalid cached/encoded embedding shape or non-finite values")
        pc.require(abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4, "Embedding must have unit norm")
        return vector

    def get_or_encode(self, texts, encode, batch_size: int = BATCH_SIZE):
        unique = sorted({text_value(text) for text in texts if text_value(text)})
        vectors, missing = {}, []
        started = perf_counter()
        for text in unique:
            path = self.path_for(text)
            if not path.exists():
                missing.append(text)
                continue
            with np.load(path, allow_pickle=False) as saved:
                pc.require(json.loads(str(saved["identity"].item())) == self.identity(text),
                           "Embedding cache identity mismatch; refusing a stale entry")
                vectors[text] = self.validate_vector(saved["vector"])
        read_seconds = perf_counter() - started
        encode_seconds, write_seconds = 0.0, 0.0
        if missing:
            started = perf_counter()
            encoded = np.asarray(encode(missing, batch_size), dtype=np.float32)
            encode_seconds = perf_counter() - started
            pc.require(encoded.shape == (len(missing), self.context.dimension), "Unexpected encoder output shape")
            started = perf_counter()
            self.directory.mkdir(parents=True, exist_ok=True)
            for text, vector in zip(missing, encoded):
                vector = self.validate_vector(vector)
                path = self.path_for(text)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, vector=vector,
                                        identity=json.dumps(self.identity(text), ensure_ascii=False, sort_keys=True))
                temporary.replace(path)
                vectors[text] = vector
            write_seconds = perf_counter() - started
        hits = len(unique) - len(missing)
        state = "empty" if not unique else "cold" if not hits else "warm" if not missing else "mixed"
        return vectors, {
            "unique_nonempty_texts": len(unique), "cache_hits": hits, "cache_misses": len(missing),
            "cache_state": state, "batch_size": batch_size,
            "cache_read_seconds": read_seconds, "embedding_encode_seconds": encode_seconds,
            "cache_write_seconds": write_seconds,
            "total_embedding_preparation_seconds": read_seconds + encode_seconds + write_seconds,
        }


def project_text_records(records):
    """Project away identifiers and every field except observed name/address text."""
    return {record_id: {field: text_value(row[field]) for field in TEXT_FIELDS}
            for record_id, row in records.items()}


def unique_texts(records):
    for row in records.values():
        pc.require(set(row) == set(TEXT_FIELDS), "Only observed names/addresses may enter embedding retrieval")
    return sorted({text_value(row[field]) for row in records.values() for field in TEXT_FIELDS
                   if text_value(row[field])})


def extend_pair_features(base: pd.DataFrame, pairs, records, vectors) -> pd.DataFrame:
    pf.validate_feature_frame(base)
    pc.require(len(base) == len(pairs), "Embedding feature row count mismatch")
    unique_texts(records)  # Validate the observed-text-only boundary, even for empty pairs.
    result = base.copy()
    for field, feature in zip(TEXT_FIELDS, EMBEDDING_NAMES):
        scores = []
        for pair in pairs:
            a = text_value(records[pair["record_id_left"]][field])
            b = text_value(records[pair["record_id_right"]][field])
            if not a or not b:
                scores.append(0.0)
                continue
            left, right = np.asarray(vectors[a], dtype=float), np.asarray(vectors[b], dtype=float)
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            pc.require(denominator > 0, "Cannot compute cosine for zero-length vectors")
            # Clip only floating-point roundoff; do not remap negative cosines to [0, 1].
            scores.append(float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)))
        result[feature] = scores
    validate_feature_frame(result)
    return result


def validate_feature_frame(frame):
    pc.require(isinstance(frame, pd.DataFrame) and tuple(frame.columns) == FEATURE_NAMES,
               "Embedding model requires the exact ordered 23-feature schema")
    pf.validate_feature_frame(frame.loc[:, list(pf.FEATURE_NAMES)])
    extra = frame.loc[:, list(EMBEDDING_NAMES)].to_numpy(dtype=float)
    pc.require(np.isfinite(extra).all() and ((-1 <= extra) & (extra <= 1)).all(),
               "Embedding cosines must be finite values in [-1, 1]")


def prepare_real_encoder(root: Path):
    """Download official weights at a pinned SHA, then load locally on CPU only."""
    cache_dir = root / ".cache/huggingface"
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import HfApi, snapshot_download
    import torch
    from sentence_transformers import SentenceTransformer

    started = perf_counter()
    pin_path = root / "models/embedding_experiment/pretrained_revision.json"
    if pin_path.exists():
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
        pc.require(pin["model_id"] == MODEL_ID, "Pinned model does not match required encoder")
        revision = pin["revision"]
    else:
        revision = HfApi(token=False).model_info(MODEL_ID, revision="main").sha
        pc.write_json(pin_path, {"model_id": MODEL_ID, "revision": revision})
    pc.require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "Expected a resolved immutable model commit SHA")
    options = {"repo_id": MODEL_ID, "revision": revision, "cache_dir": str(cache_dir), "token": False,
               "allow_patterns": ["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
               "ignore_patterns": ["onnx/*", "openvino/*", "*.bin", "*.h5", "*.msgpack"]}
    weights_state = "warm"
    try:
        snapshot = Path(snapshot_download(**options, local_files_only=True))
        pc.require((snapshot / "model.safetensors").is_file() and (snapshot / "modules.json").is_file(),
                   "Local model snapshot is incomplete")
    except (OSError, ValueError):
        weights_state = "download_required"
        snapshot = Path(snapshot_download(**options, local_files_only=False))
    pc.require(snapshot.name == revision, "Downloaded model snapshot revision mismatch")
    download_seconds = perf_counter() - started
    started = perf_counter()
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(42)
    encoder = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True,
                                  trust_remote_code=False, model_kwargs={"use_safetensors": True})
    encoder.eval()
    pc.require(encoder.device.type == "cpu", "CPU inference is required")
    pc.require(encoder.get_sentence_embedding_dimension() == 384, "Unexpected MiniLM embedding dimension")
    context = CacheContext(revision, preprocessing_version(),
                           f"sentence-transformers={version('sentence-transformers')};torch={torch.__version__}")
    return encoder, context, {
        "model_weights_cache_state": weights_state,
        "revision_resolution_and_download_seconds": download_seconds,
        "model_load_seconds": perf_counter() - started,
        "cpu_threads": CPU_THREADS, "device": "cpu", "fine_tuning": False,
    }


def encode_function(encoder):
    def encode(texts, batch_size):
        return encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False, device="cpu")
    return encode
