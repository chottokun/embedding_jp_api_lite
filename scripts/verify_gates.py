"""Run all 5 verification gates from INSTRUCTIONS.md and print numeric results."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import List

import httpx
import numpy as np
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = PROJECT_ROOT / ".hf_cache"
MODEL_ID = os.getenv("LITE_MODEL", "Chottokun/ruri-v3-30m-lite")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _start_server(port: int, offline: bool = False) -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    env["LITE_MODEL"] = MODEL_ID
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(port)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 120.0
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Server exited early ({proc.returncode}):\n{out}")
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.3)

    proc.terminate()
    raise TimeoutError(f"Server did not become ready on port {port} within 120s")


def _stop_server(proc: subprocess.Popen) -> str:
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out or ""


def main() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)

    port = _free_port()
    proc = _start_server(port, offline=False)
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Gate 1: Equivalence with direct lite loader (JA/EN x query/document)
        from app.loader import LiteEmbeddingEngine

        direct_engine = LiteEmbeddingEngine(model_id=MODEL_ID)
        direct_embedder = direct_engine.embedder

        cases = [
            ("ja_query", "日本の首都はどこですか？", "query"),
            ("ja_document", "日本の首都は東京です。", "document"),
            ("en_query", "What is the capital of Japan?", "query"),
            ("en_document", "Tokyo is the capital city of Japan.", "document"),
        ]
        cos_scores: dict[str, float] = {}
        for label, text, kind in cases:
            if kind == "query":
                ref_vec = direct_embedder.encode_query(text)[0]
            else:
                ref_vec = direct_embedder.encode_passages(text)[0]

            resp = httpx.post(
                f"{base_url}/v1/embeddings",
                json={"model": MODEL_ID, "input": text, "input_type": kind},
                timeout=30.0,
            )
            resp.raise_for_status()
            api_vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
            cos_scores[label] = _cosine(ref_vec, api_vec)
            assert abs(cos_scores[label] - 1.0) < 1e-5, f"{label} cos={cos_scores[label]}"

        print(
            "[Gate 1 PASS] Direct loader vs API cosine similarity: "
            + ", ".join(f"{k}={v:.7f}" for k, v in cos_scores.items())
        )

        # Gate 2: OpenAI SDK compatibility
        sdk_client = OpenAI(base_url=f"{base_url}/v1", api_key="sk-no-key")
        sdk_resp = sdk_client.embeddings.create(
            model=MODEL_ID,
            input=["日本語のテスト文です。", "English test sentence."],
            extra_body={"input_type": "query"},
        )
        dims = [len(d.embedding) for d in sdk_resp.data]
        assert dims == [256, 256], f"Unexpected dims: {dims}"
        assert sdk_resp.usage.prompt_tokens > 0
        print(
            f"[Gate 2 PASS] OpenAI SDK embeddings.create: count={len(sdk_resp.data)}, "
            f"dims={dims}, prompt_tokens={sdk_resp.usage.prompt_tokens}"
        )

        # Gate 5: Speed benchmark (32 short sentences on CPU)
        batch_32: List[str] = [
            f"これはCPU速度ベンチマーク用の短文テスト番号{i}です。" for i in range(32)
        ]
        # Warmup
        for _ in range(2):
            httpx.post(
                f"{base_url}/v1/embeddings",
                json={"model": MODEL_ID, "input": batch_32, "input_type": "document"},
                timeout=60.0,
            ).raise_for_status()

        runs = 5
        elapsed_list = []
        for _ in range(runs):
            t0 = time.perf_counter()
            r = httpx.post(
                f"{base_url}/v1/embeddings",
                json={"model": MODEL_ID, "input": batch_32, "input_type": "document"},
                timeout=60.0,
            )
            r.raise_for_status()
            elapsed_list.append(time.perf_counter() - t0)

        mean_sec = sum(elapsed_list) / len(elapsed_list)
        sent_per_sec = 32.0 / mean_sec
        print(
            f"[Gate 5 PASS] Batch-32 CPU latency: mean={mean_sec * 1000:.1f} ms "
            f"({sent_per_sec:.1f} sentences/sec over {runs} runs)"
        )
    finally:
        online_logs = _stop_server(proc)

    # Gate 3: Offline startup (HF_HUB_OFFLINE=1)
    offline_port = _free_port()
    offline_proc = _start_server(offline_port, offline=True)
    try:
        r_health = httpx.get(f"http://127.0.0.1:{offline_port}/health", timeout=5.0)
        r_health.raise_for_status()
        r_emb = httpx.post(
            f"http://127.0.0.1:{offline_port}/v1/embeddings",
            json={"model": MODEL_ID, "input": "オフライン起動確認", "input_type": "query"},
            timeout=30.0,
        )
        r_emb.raise_for_status()
        emb_len = len(r_emb.json()["data"][0]["embedding"])
        assert emb_len == 256
        print(
            f"[Gate 3 PASS] HF_HUB_OFFLINE=1 startup & inference: "
            f"health={r_health.json()}, embedding_dim={emb_len}"
        )
    finally:
        offline_logs = _stop_server(offline_proc)

    # Gate 4: No torch / transformers in dependency tree or loaded modules
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    forbidden = {"torch", "transformers"} & installed
    assert not forbidden, f"Forbidden packages installed: {forbidden}"
    assert "torch_or_transformers_loaded=False" in online_logs
    assert "torch_or_transformers_loaded=False" in offline_logs
    print(
        f"[Gate 4 PASS] Zero-Torch verified: installed_forbidden={sorted(forbidden)}, "
        f"packages_count={len(installed)}, startup_log_flag=torch_or_transformers_loaded=False"
    )


if __name__ == "__main__":
    main()
