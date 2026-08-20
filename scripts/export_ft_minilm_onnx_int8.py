"""Export the FT MiniLM checkpoint to ONNX int8. Do not register on failure."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

CKPT = Path("data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo")
EXPORT = Path("data/reranker_ft/ms-marco-minilm-l6-v2-ft-medlocomo-onnx")
FAILED = Path("results/finetune-reranker/onnx_export_FAILED.md")
ONNX_REL = Path("onnx") / "model_qint8_avx512.onnx"


def _paths_from_argv(argv: list[str]) -> None:
    global CKPT, EXPORT, FAILED
    if len(argv) >= 2:
        CKPT = Path(argv[1])
    if len(argv) >= 3:
        EXPORT = Path(argv[2])
    if "orpo" in CKPT.name:
        FAILED = Path("results/finetune-reranker/onnx_export_orpo_FAILED.md")


def _fail(msg: str) -> int:
    FAILED.parent.mkdir(parents=True, exist_ok=True)
    FAILED.write_text(f"# ONNX int8 export FAILED\n\n{msg}\n", encoding="utf-8")
    print(msg, file=sys.stderr)
    return 1


def main() -> int:
    _paths_from_argv(sys.argv)
    if not (CKPT / "config.json").is_file():
        return _fail(f"missing checkpoint at {CKPT}")
    EXPORT.mkdir(parents=True, exist_ok=True)
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError as exc:
        return _fail(f"optimum.onnxruntime import failed: {exc}")

    try:
        model = ORTModelForSequenceClassification.from_pretrained(str(CKPT), export=True)
        fp32_dir = EXPORT / "_fp32"
        if fp32_dir.exists():
            shutil.rmtree(fp32_dir)
        model.save_pretrained(str(fp32_dir))
        quantizer = ORTQuantizer.from_pretrained(model)
        qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
        onnx_dir = EXPORT / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(save_dir=str(onnx_dir), quantization_config=qconfig)
    except Exception as exc:  # noqa: BLE001 — export failure is a reported outcome
        return _fail(f"{type(exc).__name__}: {exc}")

    produced = list(onnx_dir.glob("*.onnx"))
    if not produced:
        return _fail(f"quantize wrote no .onnx under {onnx_dir}")
    dest = EXPORT / ONNX_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Prefer a qint8-named file if optimum produced one; else rename the first.
    src = next((p for p in produced if "qint8" in p.name or "quant" in p.name), produced[0])
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    if not dest.is_file():
        return _fail(f"expected {dest} after copy")
    # Copy tokenizer / config so CrossEncoder(local_dir) can construct.
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src_meta = CKPT / name
        if src_meta.is_file():
            shutil.copy2(src_meta, EXPORT / name)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
