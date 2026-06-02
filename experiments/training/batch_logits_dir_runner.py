"""
Evalúa una CARPETA de clips contra una lista de modelos y vuelca el logit de clase[1].

Estructura esperada de --input-dir:

    input_dir/
        <clip_1>/
            meta.json
            user_0/
                poses.npy
            user_1/
                poses.npy
        <clip_2>/
            meta.json
            user_0/
                poses.npy
        ...

Para cada poses.npy (un usuario) se ejecuta la inferencia (mismo pipeline por
defecto que test_model2.py) con cada modelo de --models, y se guarda el LOGIT
CRUDO de clase[1] (índice 1 de la salida del modelo).

Antes de evaluar se valida cada poses.npy (los no válidos se OMITEN, pero se
cuentan en el recuento). Un poses.npy es válido si:
    - No está vacío y tiene forma [T, 8, 2] (los 8 keypoints).
    - Tiene MÁS de 36 frames (3 s) con la "persona entera" visible, donde un
      frame cuenta si sus 8 keypoints son visibles (ninguno (0,0) ni NaN),
      igual que la extracción / test_model2 (un keypoint no detectado es (0,0)).

Dispositivo: --device cpu | gpu (=cuda) | auto.
Modo --single: solo evalúa clips con un único usuario (un solo poses.npy);
los clips multiusuario se omiten.

Modelos (--models): fichero JSON con una lista. Cada elemento puede ser:
    - la ruta completa al .pt (el nombre de columna = nombre del fichero):
        [
          "/ruta/completa/modelo_25.pt",
          "/ruta/completa/modelo_57.pt"
        ]
    - o un objeto con nombre explícito:
        [
          { "nombre": "modelo_25", "modelo": "/ruta/completa/modelo_25.pt" }
        ]
(También se acepta {"modelos": [...]} o {"models": [...]}.)

Salida (--output): el formato se elige por la extensión del fichero:
    - .csv  -> matriz: columnas [clip, usuario, <modelo_1>, <modelo_2>, ...].
    - .json -> { "modelos": [...], "resultados": [ {clip, usuario, muestra, <modelo>: logit, ...}, ... ] }

Uso:
    python batch_logits_dir_runner.py \
        --input-dir /ruta/a/clips \
        --models models_config.json \
        --output /ruta/resultados.csv
    python batch_logits_dir_runner.py --input-dir ... --models ... --output salida.json --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# --- Criterios de validez de un poses.npy (un usuario) ---
EXPECTED_KPS = 8          # 8 keypoints: hombros, codos, muñecas, caderas (KEEP_KPS de pose_extractor_clean)
MIN_VALID_FRAMES = 36     # "más de 36 frames" (3 s a 12 fps) con la persona entera visible

# experiments/ y experiments/training/ en el path para que test_model2 resuelva sus imports.
_THIS_DIR = Path(__file__).resolve().parent          # experiments/training
_EXPERIMENTS_DIR = _THIS_DIR.parent                  # experiments
for _p in (str(_THIS_DIR), str(_EXPERIMENTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_model2 as tm  # noqa: E402  (reutilizamos su lógica de inferencia)


def load_models_config(models_path: Path) -> List[Dict[str, str]]:
    """
    Lee la lista de modelos. Acepta lista directa o {'modelos'|'models': [...]}.
    Cada elemento puede ser:
        - una cadena con la ruta completa al .pt (el nombre de columna = nombre del fichero), o
        - un objeto {"nombre": ..., "modelo": "/ruta/completa.pt"}.
    """
    with open(models_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("modelos", data.get("models", []))
    if not isinstance(data, list) or not data:
        raise SystemExit("El fichero de modelos debe ser una lista no vacía.")
    modelos: List[Dict[str, str]] = []
    for i, m in enumerate(data, start=1):
        if isinstance(m, str):
            ruta = m.strip()
            nombre = Path(ruta).stem
        elif isinstance(m, dict):
            ruta = str(m.get("modelo", m.get("path", ""))).strip()
            nombre = str(m.get("nombre") or m.get("name") or Path(ruta).stem)
        else:
            raise SystemExit(f"Modelo #{i}: formato no válido (usa una ruta o un objeto).")
        if not ruta:
            raise SystemExit(f"Modelo #{i}: falta la ruta al .pt.")
        modelos.append({"nombre": nombre, "modelo": ruta})
    return modelos


def build_model_from_checkpoint(
    model_path: Path, device: torch.device
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """Carga el checkpoint y construye el modelo igual que test_model2.main()."""
    checkpoint = torch.load(str(model_path), map_location="cpu")
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(checkpoint["label_to_idx"])))
    model = tm.build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def discover_samples(input_dir: Path, single_only: bool = False) -> List[Tuple[str, str, Path]]:
    """
    Recorre input_dir/<clip>/<usuario>/poses.npy.

    Devuelve lista de (clip_name, usuario, user_dir). clip_name viene de meta.json
    si existe, si no del nombre de la carpeta del clip.

    Si single_only=True, solo se incluyen los clips con EXACTAMENTE un usuario
    (un único poses.npy); los clips multiusuario se omiten.
    """
    samples: List[Tuple[str, str, Path]] = []
    for clip_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        clip_name = clip_dir.name
        meta_p = clip_dir / "meta.json"
        if meta_p.exists():
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                clip_name = str(meta.get("clip_name", clip_name))
            except Exception:
                pass
        else:
            print(f"[WARN] '{clip_dir.name}' no tiene meta.json (se usa el nombre de carpeta).")

        user_dirs = [
            ud for ud in sorted(clip_dir.iterdir())
            if ud.is_dir() and (ud / "poses.npy").is_file()
        ]
        if not user_dirs:
            print(f"[WARN] '{clip_dir.name}' no tiene carpetas de usuario con poses.npy. Se omite.")
            continue
        if single_only and len(user_dirs) != 1:
            print(f"[SKIP] '{clip_dir.name}' tiene {len(user_dirs)} usuarios (modo --single: solo 1). Se omite.")
            continue
        for ud in user_dirs:
            samples.append((clip_name, ud.name, ud))
    return samples


def validate_poses_npy(poses_path: Path) -> Tuple[bool, str, int, int]:
    """
    Comprueba si un poses.npy es válido para evaluar.

    Criterios (igual que la extracción / test_model2: un keypoint no detectado es (0,0)):
        - No vacío y con forma [T, 8, 2] (los 8 keypoints).
        - Frame con "persona entera" = los 8 keypoints visibles (ninguno (0,0) ni NaN).
        - Más de MIN_VALID_FRAMES (36) frames con la persona entera visible.

    Devuelve (es_valido, motivo, n_frames_validos, n_frames_totales).
    """
    try:
        arr = np.load(str(poses_path), allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        return False, f"no_se_pudo_leer ({e})", 0, 0

    if getattr(arr, "size", 0) == 0 or getattr(arr, "ndim", 0) == 0:
        return False, "vacio", 0, 0
    if arr.ndim != 3 or int(arr.shape[2]) != 2:
        return False, f"shape_invalida {tuple(arr.shape)} (se esperaba [T,8,2])", 0, int(arr.shape[0]) if arr.ndim >= 1 else 0
    t_total = int(arr.shape[0])
    if t_total == 0:
        return False, "vacio (0 frames)", 0, 0
    if int(arr.shape[1]) != EXPECTED_KPS:
        return False, f"no_tiene_8_keypoints (J={int(arr.shape[1])})", 0, t_total

    arr = arr.astype(np.float64, copy=False)
    is_zero = (arr[..., 0] == 0.0) & (arr[..., 1] == 0.0)          # [T, 8] keypoint ausente
    is_nan = np.isnan(arr).any(axis=2)                             # [T, 8] keypoint NaN
    not_visible = is_zero | is_nan                                 # [T, 8]
    frame_full_body = ~not_visible.any(axis=1)                     # [T] los 8 visibles
    n_valid = int(frame_full_body.sum())

    if n_valid <= MIN_VALID_FRAMES:
        return False, f"pocos_frames_validos ({n_valid} <= {MIN_VALID_FRAMES})", n_valid, t_total
    return True, "ok", n_valid, t_total


def _value_from_logits(logits_1d: "torch.Tensor", value_kind: str) -> float:
    """
    Calcula el valor de salida a partir del vector de logits [num_clases]:
        - logit1  : logit crudo de clase[1] (por defecto).
        - logit0  : logit crudo de clase[0].
        - margin  : logit[1] - logit[0] (margen del argmax, "confianza" de robo).
        - prob1   : softmax -> P(clase 1) en [0,1].
        - entropy : entropía de la softmax (incertidumbre; alta = poco fiable).
    """
    if int(logits_1d.shape[0]) < 2:
        raise RuntimeError(
            f"El modelo solo produjo {int(logits_1d.shape[0])} logit(s); no existe clase[1]."
        )
    if value_kind == "logit1":
        return float(logits_1d[1].item())
    if value_kind == "logit0":
        return float(logits_1d[0].item())
    if value_kind == "margin":
        return float((logits_1d[1] - logits_1d[0]).item())
    probs = torch.softmax(logits_1d, dim=0)
    if value_kind == "prob1":
        return float(probs[1].item())
    if value_kind == "entropy":
        p = probs.clamp_min(1e-12)
        return float((-(p * p.log()).sum()).item())
    raise ValueError(f"value_kind desconocido: {value_kind}")


def logit_class1(
    *,
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    user_dir: Path,
    clip_name: str,
    device: torch.device,
    tmp_dir: Path,
    value_kind: str = "logit1",
) -> float:
    """Inferencia (pipeline por defecto de test_model2) sobre un user_dir y valor de salida elegido."""
    _tid, _prob, logits, _clf_ms, _total_ms, cleanup = tm.infer_user_track(
        checkpoint=checkpoint,
        model=model,
        user_dir=user_dir,
        pose_source="filtered",
        clip_name=clip_name,
        device=device,
        simple_preprocess=False,
        use_temp_npy=True,
        tmp_dir=tmp_dir,
    )
    tm._safe_unlink(cleanup)
    logits_1d = logits.detach().float().cpu().reshape(-1)
    return _value_from_logits(logits_1d, value_kind)


def write_csv(output_path: Path, model_names: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "usuario"] + model_names)
        for r in rows:
            writer.writerow(
                [r["clip"], r["usuario"]] + [r["logits"].get(m, "") for m in model_names]
            )


def write_json(output_path: Path, model_names: List[str], rows: List[Dict[str, Any]]) -> None:
    resultados = []
    for r in rows:
        rec: Dict[str, Any] = {
            "clip": r["clip"],
            "usuario": r["usuario"],
            "muestra": f"{r['clip']}/{r['usuario']}",
        }
        rec.update({m: r["logits"].get(m) for m in model_names})
        resultados.append(rec)
    payload = {"modelos": model_names, "resultados": resultados}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa una carpeta de clips (user_*/poses.npy) contra varios modelos -> logit clase[1]."
    )
    parser.add_argument("--input-dir", required=True, help="Carpeta raíz con subcarpetas de clips.")
    parser.add_argument("--models", required=True, help="JSON con la lista de modelos (nombre + ruta .pt).")
    parser.add_argument("--output", required=True, help="Ruta de salida (.csv o .json).")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cuda", "auto"],
        default="auto",
        help="cpu | gpu (=cuda) | auto (autodetección, por defecto).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Solo evalúa clips con un único usuario (un solo poses.npy); omite los multiusuario.",
    )
    parser.add_argument(
        "--value",
        choices=["logit1", "logit0", "margin", "prob1", "entropy"],
        default="logit1",
        help=(
            "Valor a guardar por celda: logit1 (def, logit clase[1]), logit0, "
            "margin (logit[1]-logit[0]), prob1 (softmax P(robo)), entropy (incertidumbre)."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    models_path = Path(args.models).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    out_fmt = output_path.suffix.lower()
    if out_fmt not in (".csv", ".json"):
        raise SystemExit("--output debe terminar en .csv o .json")

    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir no es una carpeta: {input_dir}")
    if not models_path.is_file():
        raise SystemExit(f"No existe el fichero de modelos: {models_path}")

    if args.device in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("Se pidió GPU/CUDA pero torch.cuda.is_available() es False.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Valor por celda: {args.value}")

    modelos = load_models_config(models_path)
    model_names = [m["nombre"] for m in modelos]
    if len(set(model_names)) != len(model_names):
        raise SystemExit("Hay nombres de modelo repetidos en --models; deben ser únicos (son columnas).")

    # --- Comprobación previa: existen todos los modelos y hay muestras válidas ---
    errores: List[str] = []
    for m in modelos:
        if not Path(m["modelo"]).expanduser().resolve().is_file():
            errores.append(f"No existe el modelo '{m['nombre']}': {m['modelo']}")

    if args.single:
        print("[INFO] Modo --single: solo clips con un único usuario.")
    all_samples = discover_samples(input_dir, single_only=args.single)
    if not all_samples:
        errores.append(f"No se encontró ningún 'poses.npy' bajo {input_dir} (estructura clip/usuario/poses.npy).")

    if errores:
        print(f"[ABORTADO] Se encontraron {len(errores)} problema(s):")
        for msg in errores:
            print(f"  - {msg}")
        raise SystemExit(1)

    # --- Validación de cada poses.npy + recuento ---
    print(f"[INFO] poses.npy encontrados: {len(all_samples)}. Validando...")
    samples: List[Tuple[str, str, Path]] = []
    invalidos: List[Tuple[str, str, str]] = []
    for clip_name, usuario, user_dir in all_samples:
        ok, motivo, n_valid, t_total = validate_poses_npy(user_dir / "poses.npy")
        if ok:
            samples.append((clip_name, usuario, user_dir))
        else:
            invalidos.append((clip_name, usuario, motivo))
            print(f"[SKIP] {clip_name}/{usuario}: {motivo} (frames_validos={n_valid}, T={t_total})")

    print("[RECUENTO] -----------------------------------------")
    print(f"[RECUENTO] npy totales encontrados : {len(all_samples)}")
    print(f"[RECUENTO] npy válidos             : {len(samples)}")
    print(f"[RECUENTO] npy descartados         : {len(invalidos)}")
    if invalidos:
        motivos = Counter(m.split(" ")[0].split("(")[0] for _, _, m in invalidos)
        for mot, cnt in motivos.most_common():
            print(f"[RECUENTO]   - {mot}: {cnt}")
    print("[RECUENTO] -----------------------------------------")

    if not samples:
        raise SystemExit("No hay ningún poses.npy válido que evaluar (todos descartados).")

    print(f"[INFO] Modelos: {len(modelos)} | Muestras válidas: {len(samples)}")
    print(f"[INFO] Total inferencias a realizar: {len(modelos) * len(samples)}")
    print("[INFO] Comprobación previa OK.")

    # Carga (una vez) cada modelo.
    loaded: List[Tuple[str, Dict[str, Any], torch.nn.Module]] = []
    for m in modelos:
        ckpt, model = build_model_from_checkpoint(Path(m["modelo"]).expanduser().resolve(), device)
        loaded.append((m["nombre"], ckpt, model))

    tmp_root = Path(tempfile.mkdtemp(prefix="batch_logits_dir_"))
    rows: List[Dict[str, Any]] = []
    try:
        for si, (clip_name, usuario, user_dir) in enumerate(samples, start=1):
            row: Dict[str, Any] = {"clip": clip_name, "usuario": usuario, "logits": {}}
            print(f"[INFO] ({si}/{len(samples)}) clip='{clip_name}' usuario='{usuario}'")
            for mname, ckpt, model in loaded:
                try:
                    val = logit_class1(
                        checkpoint=ckpt,
                        model=model,
                        user_dir=user_dir,
                        clip_name=clip_name,
                        device=device,
                        tmp_dir=tmp_root,
                        value_kind=args.value,
                    )
                    row["logits"][mname] = val
                    print(f"    [{mname}] logit clase[1] = {val:+.6f}")
                except Exception as e:  # noqa: BLE001
                    row["logits"][mname] = f"ERROR:{type(e).__name__}"
                    print(f"    [{mname}] ERROR: {e}")
            rows.append(row)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if out_fmt == ".csv":
        write_csv(output_path, model_names, rows)
    else:
        write_json(output_path, model_names, rows)

    print(f"[INFO] Resultado escrito en: {output_path}")
    print(f"[INFO] Filas (clip/usuario): {len(rows)} | Columnas (modelos): {len(model_names)}")


if __name__ == "__main__":
    main()
