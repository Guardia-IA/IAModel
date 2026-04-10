"""
Valida rangos de operaciones sobre un .npy de poses sin sacar el esqueleto de ventana.

Para cada operacion reporta:
- rango estricto: mantiene todos los puntos dentro de [0,1] sin ayuda adicional
- rango compensable: puede salirse, pero seria recolocable completo con un shift global
  (equivale a que el ancho/alto de la nube de puntos transformada sea <= 1)

Uso:
    python validate_npy.py /ruta/poses.npy --step 0.01
    python validate_npy.py /ruta/poses.npy --step 0.1 --angle-min -180 --angle-max 180
"""

import json
import argparse
import math
import os
from typing import Any, List, Tuple

import numpy as np


def _decimals_from_step(step: float) -> int:
    s = f"{step:.12f}".rstrip("0").rstrip(".")
    if "." not in s:
        return 0
    return max(0, len(s.split(".")[1]))


def _round_to(v: float, decimals: int) -> float:
    return float(round(v, decimals))


def _fmt(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}"


def _intervals_from_values(values: List[float], step: float, tol: float = 1e-9) -> List[Tuple[float, float]]:
    if not values:
        return []
    vals = sorted(values)
    intervals: List[Tuple[float, float]] = []
    start = vals[0]
    prev = vals[0]
    for v in vals[1:]:
        if abs(v - prev - step) <= max(tol, step * 1e-6):
            prev = v
            continue
        intervals.append((start, prev))
        start = v
        prev = v
    intervals.append((start, prev))
    return intervals


def _check_shape(data: np.ndarray) -> None:
    if data.ndim < 3 or data.shape[-1] < 2:
        raise ValueError(
            f"Formato no soportado: {data.shape}. "
            "Esperado algo como (T, J, 2) o (T, U, J, 2)."
        )


def _extract_xy(data: np.ndarray) -> np.ndarray:
    # Aplana cualquier shape [..., 2] a [N, 2]
    return data[..., :2].reshape(-1, 2).astype(np.float64)


def _scale_points(xy: np.ndarray, factor: float) -> np.ndarray:
    out = xy.copy()
    out[:, 0] = (out[:, 0] - 0.5) * factor + 0.5
    out[:, 1] = (out[:, 1] - 0.5) * factor + 0.5
    return out


def _rotate_points(xy: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    x = xy[:, 0] - 0.5
    y = xy[:, 1] - 0.5
    xr = x * c - y * s
    yr = x * s + y * c
    out = np.empty_like(xy)
    out[:, 0] = xr + 0.5
    out[:, 1] = yr + 0.5
    return out


def _in_bounds_01(xy: np.ndarray) -> bool:
    return bool(np.all((xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)))


def _compensable_by_shift(xy: np.ndarray) -> bool:
    # Existe dx,dy para meter todo en [0,1] si ancho<=1 y alto<=1
    w = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    h = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    return w <= 1.0 + 1e-12 and h <= 1.0 + 1e-12


def _step_bounds(lo: float, hi: float, step: float) -> Tuple[float | None, float | None]:
    if step <= 0:
        return None, None
    first = math.ceil(lo / step) * step
    last = math.floor(hi / step) * step
    if first > last:
        return None, None
    return first, last


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_profile(cfg: dict, profile: str) -> dict:
    if not isinstance(cfg, dict):
        return {}
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict):
        return cfg
    selected = profiles.get(profile)
    if selected is None:
        available = ", ".join(sorted(profiles.keys())) if profiles else "(ninguno)"
        raise ValueError(f"Perfil '{profile}' no encontrado. Disponibles: {available}")
    if not isinstance(selected, dict):
        raise ValueError(f"Perfil '{profile}' inválido en JSON.")
    return selected


def _count_discrete_values(lo: float, hi: float, step: float) -> int:
    if step <= 0:
        return 0
    first, last = _step_bounds(lo, hi, step)
    if first is None or last is None:
        return 0
    return int(math.floor((last - first) / step + 1e-9)) + 1


def _grid_values(lo: float, hi: float, step: float) -> List[float]:
    first, last = _step_bounds(lo, hi, step)
    if first is None or last is None:
        return []
    n = _count_discrete_values(lo, hi, step)
    return [first + i * step for i in range(n)]


def _estimate_optimal_n(features: np.ndarray) -> Tuple[int, int, float]:
    """
    Estima cuántas variantes seleccionar maximizando diversidad y minimizando redundancia.
    Devuelve:
      - n_opt (punto recomendado)
      - n_total
      - cobertura normalizada alcanzada en n_opt (0..1)
    """
    n_total = int(features.shape[0])
    if n_total <= 2:
        return n_total, n_total, 1.0

    # Distancias euclídeas en espacio de features normalizado.
    diff = features[:, None, :] - features[None, :, :]
    dmat = np.sqrt(np.sum(diff * diff, axis=2))
    dmax = float(np.max(dmat))
    if dmax <= 1e-12:
        return 1, n_total, 1.0

    # Farthest-point sampling (codicioso) para cubrir diversidad.
    selected = [0]
    min_dist = dmat[0].copy()
    gains: List[float] = [float(np.mean(min_dist / dmax))]
    coverages: List[float] = [gains[0]]
    while len(selected) < n_total:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, dmat[idx])
        cov = float(np.mean(min_dist / dmax))
        coverages.append(cov)
        gains.append(max(0.0, coverages[-2] - coverages[-1]))

    # Detecta "codo": cuando la mejora marginal cae de forma sostenida.
    first_gain = gains[1] if len(gains) > 1 else 0.0
    if first_gain <= 1e-12:
        return min(10, n_total), n_total, 1.0 - coverages[min(9, len(coverages) - 1)]
    threshold = first_gain * 0.15
    patience = 6
    streak = 0
    n_opt = n_total
    for i in range(2, len(gains)):
        if gains[i] < threshold:
            streak += 1
        else:
            streak = 0
        if streak >= patience:
            n_opt = max(8, i - patience + 1)
            break
    cov_at_opt = 1.0 - coverages[min(n_opt - 1, len(coverages) - 1)]
    return n_opt, n_total, cov_at_opt


def _farthest_point_order(features: np.ndarray) -> List[int]:
    n_total = int(features.shape[0])
    if n_total == 0:
        return []
    if n_total == 1:
        return [0]
    diff = features[:, None, :] - features[None, :, :]
    dmat = np.sqrt(np.sum(diff * diff, axis=2))
    order = [0]
    min_dist = dmat[0].copy()
    while len(order) < n_total:
        idx = int(np.argmax(min_dist))
        order.append(idx)
        min_dist = np.minimum(min_dist, dmat[idx])
    return order


def _industrial_target_n(n_total: int, n_min: int, coverage: float) -> int:
    """
    Devuelve un N objetivo para uso industrial:
    - respeta el mínimo conservador
    - prioriza rango práctico 20-80
    - crece con nº de candidatas y cobertura
    """
    if n_total <= 0:
        return 0
    # Base por tamaño (sublineal) y diversidad.
    # coverage en [0,1], más cobertura -> más capacidad de explotar variantes.
    base = 8.0 + 5.0 * math.sqrt(float(n_total))
    factor_cov = 0.55 + 0.45 * max(0.0, min(1.0, coverage))
    target = int(round(base * factor_cov))
    target = max(20, min(80, target))
    target = max(n_min, target)
    target = min(n_total, target)
    return target


def _write_manifest(
    manifest_path: str,
    npy_path: str,
    profile: str,
    n_total: int,
    n_min: int,
    n_industrial: int,
    coverage: float,
    candidates: List[dict[str, Any]],
    selected_min: List[dict[str, Any]],
    selected_industrial: List[dict[str, Any]],
    selected_industrial_with_mirror: List[dict[str, Any]],
    selected_industrial_compose_light: List[dict[str, Any]],
    on_the_fly_policy: dict[str, Any],
) -> None:
    payload = {
        "source_npy": npy_path,
        "profile": profile,
        "summary": {
            "candidates_total": n_total,
            "n_min": n_min,
            "n_objetivo_industrial": n_industrial,
            "coverage_at_n_min": coverage,
        },
        "selected_n_min": selected_min,
        "selected_n_objetivo_industrial": selected_industrial,
        "selected_n_objetivo_industrial_with_mirror_composed": selected_industrial_with_mirror,
        "selected_n_objetivo_industrial_compose_light": selected_industrial_compose_light,
        "on_the_fly_policy_recommended": on_the_fly_policy,
        "all_candidates": candidates,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def validate_npy(
    npy_path: str,
    profile_name: str,
    manifest_path: str,
    mirror_compose_ratio: float,
    compose_light_ratio: float,
    global_step: float,
    rotate_step: float,
    scale_step: float,
    shift_step: float,
    noise_step: float,
    angle_min: float,
    angle_max: float,
    noise_sigma_cap: float,
) -> None:
    data = np.load(npy_path)
    _check_shape(data)
    xy = _extract_xy(data)

    dec_rot = _decimals_from_step(rotate_step)
    dec_scale = _decimals_from_step(scale_step)
    dec_shift = _decimals_from_step(shift_step)
    dec_noise = _decimals_from_step(noise_step)
    x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    width = x_max - x_min
    height = y_max - y_min

    print(f"Archivo: {npy_path}")
    print(f"Shape: {data.shape}")
    print(f"Rango base x=[{_fmt(x_min,4)},{_fmt(x_max,4)}], y=[{_fmt(y_min,4)},{_fmt(y_max,4)}]")
    print()

    # 1) MIRROR
    print("=== mirror ===")
    print("- Estricto: siempre valido (x -> 1-x, mantiene [0,1]).")
    print("- Compensable con shift: siempre valido.")
    print()

    # 2) SHIFT (exacto)
    dx_min = -x_min
    dx_max = 1.0 - x_max
    dy_min = -y_min
    dy_max = 1.0 - y_max
    print("=== shift ===")
    print(f"- Estricto continuo dx in [{_fmt(dx_min,dec_shift)},{_fmt(dx_max,dec_shift)}]")
    print(f"- Estricto continuo dy in [{_fmt(dy_min,dec_shift)},{_fmt(dy_max,dec_shift)}]")
    sdx0, sdx1 = _step_bounds(dx_min, dx_max, shift_step)
    sdy0, sdy1 = _step_bounds(dy_min, dy_max, shift_step)
    if sdx0 is not None and sdy0 is not None:
        print(
            f"- Con step shift={shift_step}: dx in [{_fmt(sdx0,dec_shift)},{_fmt(sdx1,dec_shift)}], "
            f"dy in [{_fmt(sdy0,dec_shift)},{_fmt(sdy1,dec_shift)}]"
        )
    shift_count_x = _count_discrete_values(dx_min, dx_max, shift_step)
    shift_count_y = _count_discrete_values(dy_min, dy_max, shift_step)
    shift_total = shift_count_x * shift_count_y
    print(f"- Total variantes shift (malla dx*dy): {shift_count_x} * {shift_count_y} = {shift_total}")
    print("- Compensable con shift: no aplica (la operacion ya es shift).")
    print()

    # 3) SCALE (exacto aprox para f>=0)
    bounds = []
    for v in xy[:, 0]:
        if v > 0.5:
            bounds.append(0.5 / (v - 0.5))
        elif v < 0.5:
            bounds.append(0.5 / (0.5 - v))
    for v in xy[:, 1]:
        if v > 0.5:
            bounds.append(0.5 / (v - 0.5))
        elif v < 0.5:
            bounds.append(0.5 / (0.5 - v))
    f_max_strict = min(bounds) if bounds else 1.0
    f_max_comp = min(1.0 / max(width, 1e-12), 1.0 / max(height, 1e-12))
    print("=== scale ===")
    print(f"- Estricto continuo factor in [0,{_fmt(f_max_strict,dec_scale)}]  => porcentaje [0,{_fmt(100*f_max_strict,dec_scale)}]")
    print(
        f"- Compensable continuo factor in [0,{_fmt(f_max_comp,dec_scale)}]  => porcentaje [0,{_fmt(100*f_max_comp,dec_scale)}]"
    )
    sf0, sf1 = _step_bounds(0.0, f_max_strict, scale_step / 100.0)
    scale_count = _count_discrete_values(0.0, f_max_strict, scale_step / 100.0)
    if sf0 is not None:
        print(
            f"- Con step scale={scale_step}%: porcentaje in "
            f"[{_fmt(100*sf0,dec_scale)},{_fmt(100*sf1,dec_scale)}] (estricto)"
        )
        print(f"- Total variantes scale (estricto): {scale_count}")
    print(
        "  (compensable = podria recolocarse con un shift global para quedar totalmente visible)"
    )
    print()

    # 4) ROTATE (barrido)
    if angle_max < angle_min:
        angle_min, angle_max = angle_max, angle_min
    angle_vals = np.arange(angle_min, angle_max + rotate_step * 0.5, rotate_step, dtype=np.float64)
    strict_ok = []
    comp_ok = []
    for a in angle_vals:
        rot = _rotate_points(xy, float(a))
        if _in_bounds_01(rot):
            strict_ok.append(float(a))
        if _compensable_by_shift(rot):
            comp_ok.append(float(a))

    strict_intervals = _intervals_from_values(strict_ok, rotate_step)
    comp_intervals = _intervals_from_values(comp_ok, rotate_step)
    print("=== rotate ===")
    if strict_ok:
        print(
            f"- Estricto angulo min/max: [{_fmt(min(strict_ok),dec_rot)},{_fmt(max(strict_ok),dec_rot)}] "
            f"(step={rotate_step})"
        )
        print("  Intervalos estrictos:")
        for a0, a1 in strict_intervals:
            print(f"    [{_fmt(a0,dec_rot)}, {_fmt(a1,dec_rot)}]")
        rotate_count = len(strict_ok)
        print(f"- Total variantes rotate (estricto): {rotate_count}")
    else:
        print("- Estricto: no hay angulos validos en el barrido dado.")
        rotate_count = 0

    if comp_ok:
        print(
            f"- Compensable min/max: [{_fmt(min(comp_ok),dec_rot)},{_fmt(max(comp_ok),dec_rot)}] "
            f"(step={rotate_step})"
        )
        print("  Intervalos compensables:")
        for a0, a1 in comp_intervals:
            print(f"    [{_fmt(a0,dec_rot)}, {_fmt(a1,dec_rot)}]")
    else:
        print("- Compensable: no hay angulos validos en el barrido dado.")
    print()

    # 5) NOISE (recomendación)
    margin_x = min(x_min, 1.0 - x_max)
    margin_y = min(y_min, 1.0 - y_max)
    sigma_safe_x_3s = max(0.0, margin_x / 3.0)
    sigma_safe_y_3s = max(0.0, margin_y / 3.0)
    sigma_safe_x_2s = max(0.0, margin_x / 2.0)
    sigma_safe_y_2s = max(0.0, margin_y / 2.0)
    print("=== noise ===")
    print("- Estricto matematico: no acotable (ruido gaussiano tiene soporte infinito).")
    print(
        f"- Recomendacion 3-sigma (muy conservadora): "
        f"sigma_x <= {_fmt(sigma_safe_x_3s,dec_noise)}, sigma_y <= {_fmt(sigma_safe_y_3s,dec_noise)}"
    )
    print(
        f"- Recomendacion 2-sigma (menos conservadora): "
        f"sigma_x <= {_fmt(sigma_safe_x_2s,dec_noise)}, sigma_y <= {_fmt(sigma_safe_y_2s,dec_noise)}"
    )
    if noise_step > 0:
        sx_vals = np.arange(0.0, noise_sigma_cap + noise_step * 0.5, noise_step)
        sy_vals = np.arange(0.0, noise_sigma_cap + noise_step * 0.5, noise_step)
        sx_safe = [v for v in sx_vals if v <= sigma_safe_x_3s + 1e-12]
        sy_safe = [v for v in sy_vals if v <= sigma_safe_y_3s + 1e-12]
        if sx_safe and sy_safe:
            noise_count_x = len(sx_safe)
            noise_count_y = len(sy_safe)
            noise_total = noise_count_x * noise_count_y
            print(
                f"- Con step noise={noise_step}: "
                f"sigma_x in [0,{_fmt(max(sx_safe),dec_noise)}], "
                f"sigma_y in [0,{_fmt(max(sy_safe),dec_noise)}] (criterio 3-sigma)"
            )
            print(
                f"- Total variantes noise (malla sigma_x*sigma_y, 3-sigma): "
                f"{noise_count_x} * {noise_count_y} = {noise_total}"
            )
        else:
            noise_total = 0
            noise_count_x = 0
            noise_count_y = 0
    else:
        noise_total = 0
        noise_count_x = 0
        noise_count_y = 0
    print()

    # Resumen de cardinalidad discreta de transformaciones
    # mirror tiene 2 estados posibles: aplicar o no aplicar.
    mirror_count = 2
    total_producto = mirror_count * max(1, rotate_count) * max(1, scale_count) * max(1, shift_total) * max(1, noise_total)
    total_suma = mirror_count + rotate_count + scale_count + shift_total + noise_total
    print("=== conteo total de operaciones (según steps) ===")
    print(f"- mirror: {mirror_count} estados (sin/aplicado)")
    print(f"- rotate (estricto): {rotate_count}")
    print(f"- scale (estricto): {scale_count}")
    print(f"- shift (estricto dx*dy): {shift_total}")
    print(f"- noise (3-sigma, sigma_x*sigma_y): {noise_total}")
    print(
        "- TOTAL combinaciones (producto cartesiano, incluyendo mirror on/off): "
        f"{total_producto}"
    )
    print(
        "- TOTAL variantes individuales (sumando cada operación por separado): "
        f"{total_suma}"
    )
    print(
        "  Nota: el total por producto es una estimación de NPYs distintos si compones "
        "pipeline mirror+rotate+scale+shift+noise."
    )
    print()

    # Estimación de N óptimo (variantes individuales) para reducir redundancia.
    # Features en espacio de parámetros normalizado por operación.
    # No expande combinaciones cartesianas: trabaja sobre variantes individuales.
    dx_vals = _grid_values(dx_min, dx_max, shift_step)
    dy_vals = _grid_values(dy_min, dy_max, shift_step)
    scale_vals = _grid_values(0.0, f_max_strict, scale_step / 100.0)
    rotate_vals = strict_ok[:] if strict_ok else []
    sx_vals = _grid_values(0.0, min(noise_sigma_cap, sigma_safe_x_3s), noise_step)
    sy_vals = _grid_values(0.0, min(noise_sigma_cap, sigma_safe_y_3s), noise_step)

    rot_span = max(1e-9, max(abs(angle_min), abs(angle_max)))
    scale_span = max(1e-9, f_max_strict)
    shift_x_span = max(1e-9, max(abs(dx_min), abs(dx_max)))
    shift_y_span = max(1e-9, max(abs(dy_min), abs(dy_max)))
    noise_x_span = max(1e-9, min(noise_sigma_cap, sigma_safe_x_3s))
    noise_y_span = max(1e-9, min(noise_sigma_cap, sigma_safe_y_3s))

    feats: List[List[float]] = []
    # one-hot de operación (mirror, rotate, scale, shift, noise) + parámetros normalizados
    # mirror (solo variante aplicada; identidad no aporta diversidad)
    feats.append([1, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for a in rotate_vals:
        feats.append([0, 1, 0, 0, 0, 0.0, a / rot_span, 0.0, 0.0, 0.0, 0.0])
    for f in scale_vals:
        feats.append([0, 0, 1, 0, 0, 0.0, 0.0, f / scale_span, 0.0, 0.0, 0.0])
    for dx in dx_vals:
        for dy in dy_vals:
            feats.append([0, 0, 0, 1, 0, 0.0, 0.0, 0.0, dx / shift_x_span, dy / shift_y_span, 0.0])
    for sx in sx_vals:
        for sy in sy_vals:
            feats.append([0, 0, 0, 0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (sx / noise_x_span + sy / noise_y_span)])

    if feats:
        feat_arr = np.asarray(feats, dtype=np.float64)
        n_opt, n_total_individual, cov = _estimate_optimal_n(feat_arr)
        n_low = max(8, int(round(n_opt * 0.8)))
        n_high = min(n_total_individual, int(round(n_opt * 1.2)))
        n_industrial = _industrial_target_n(n_total_individual, n_opt, cov)

        # Selección explícita de variantes menos redundantes (FPS).
        candidates: List[dict[str, Any]] = []
        candidates.append({"operation": "mirror", "params": {"apply": True}})
        for a in rotate_vals:
            candidates.append({"operation": "rotate", "params": {"degrees": float(a)}})
        for f in scale_vals:
            candidates.append({"operation": "scale", "params": {"percentage": float(100.0 * f)}})
        for dx in dx_vals:
            for dy in dy_vals:
                candidates.append({"operation": "shift", "params": {"dx": float(dx), "dy": float(dy)}})
        for sx in sx_vals:
            for sy in sy_vals:
                candidates.append({"operation": "noise", "params": {"sigma_x": float(sx), "sigma_y": float(sy)}})

        order = _farthest_point_order(feat_arr)
        selected_min_idx = order[:n_opt]
        selected_ind_idx = order[:n_industrial]
        selected_min = [candidates[i] for i in selected_min_idx]
        selected_ind = [candidates[i] for i in selected_ind_idx]

        # Composición opcional con mirror sobre variantes industriales seleccionadas.
        # No se compone sobre operación mirror para evitar duplicado directo.
        compose_count = int(round(len(selected_ind) * mirror_compose_ratio))
        compose_count = max(0, min(len(selected_ind), compose_count))
        composable_idx = [i for i, item in enumerate(selected_ind) if item.get("operation") != "mirror"]
        compose_idx = composable_idx[:compose_count]
        selected_ind_with_mirror: List[dict[str, Any]] = []
        for i, item in enumerate(selected_ind):
            selected_ind_with_mirror.append(item)
            if i in compose_idx:
                selected_ind_with_mirror.append(
                    {
                        "operation": "compose",
                        "params": {
                            "pipeline": [
                                {"operation": item["operation"], "params": item["params"]},
                                {"operation": "mirror", "params": {"apply": True}},
                            ]
                        },
                    }
                )

        # Composición light (máximo 2 operaciones en pipeline) para robustez comercial.
        # Añade una segunda operación suave y distinta de la base.
        def _safe_mid(values: List[float], default: float) -> float:
            if not values:
                return default
            vals = sorted(values)
            return float(vals[len(vals) // 2])

        default_second_ops = [
            {"operation": "rotate", "params": {"degrees": _safe_mid(rotate_vals, 3.0)}},
            {"operation": "scale", "params": {"percentage": float(100.0 * _safe_mid(scale_vals, 1.03))}},
            {"operation": "shift", "params": {"dx": _safe_mid(dx_vals, 0.01), "dy": _safe_mid(dy_vals, 0.01)}},
            {"operation": "noise", "params": {"sigma_x": _safe_mid(sx_vals, 0.002), "sigma_y": _safe_mid(sy_vals, 0.002)}},
            {"operation": "mirror", "params": {"apply": True}},
        ]

        base_for_compose = [item for item in selected_ind if item.get("operation") != "compose"]
        compose_light_target = int(round(len(base_for_compose) * compose_light_ratio))
        compose_light_target = max(0, min(len(base_for_compose), compose_light_target))
        selected_ind_compose_light: List[dict[str, Any]] = []
        for i, item in enumerate(base_for_compose[:compose_light_target]):
            base_op = item.get("operation")
            second = next((op for op in default_second_ops if op["operation"] != base_op), None)
            if second is None:
                continue
            selected_ind_compose_light.append(
                {
                    "operation": "compose",
                    "params": {
                        "pipeline": [
                            {"operation": base_op, "params": item.get("params", {})},
                            second,
                        ]
                    },
                }
            )

        # Política recomendada de augment on-the-fly (para integrar después en train_model).
        # Objetivo: robustez comercial evitando sobre-augmentación.
        policy = {
            "version": 1,
            "note": "Recomendación automática derivada de validate_npy.py",
            "apply_augmentation_prob": 0.65,
            "max_ops_per_sample": 2,
            "probabilities": {
                "mirror": 0.30,
                "rotate": 0.28,
                "scale": 0.16,
                "shift": 0.16,
                "noise": 0.10,
            },
            "ranges": {
                "rotate_degrees": [
                    float(min(rotate_vals)) if rotate_vals else float(angle_min),
                    float(max(rotate_vals)) if rotate_vals else float(angle_max),
                ],
                "scale_percentage": [
                    float(100.0 * min(scale_vals)) if scale_vals else 100.0,
                    float(100.0 * max(scale_vals)) if scale_vals else 100.0,
                ],
                "shift_dx": [
                    float(min(dx_vals)) if dx_vals else 0.0,
                    float(max(dx_vals)) if dx_vals else 0.0,
                ],
                "shift_dy": [
                    float(min(dy_vals)) if dy_vals else 0.0,
                    float(max(dy_vals)) if dy_vals else 0.0,
                ],
                "noise_sigma_x": [
                    float(min(sx_vals)) if sx_vals else 0.0,
                    float(max(sx_vals)) if sx_vals else 0.0,
                ],
                "noise_sigma_y": [
                    float(min(sy_vals)) if sy_vals else 0.0,
                    float(max(sy_vals)) if sy_vals else 0.0,
                ],
            },
            "sampling": {
                "distribution": "uniform",
                "preserve_original_prob": 0.35,
                "compose_mirror_prob_when_augmented": mirror_compose_ratio,
                "compose_light_prob_when_augmented": compose_light_ratio,
            },
        }

        _write_manifest(
            manifest_path=manifest_path,
            npy_path=npy_path,
            profile=profile_name,
            n_total=n_total_individual,
            n_min=n_opt,
            n_industrial=n_industrial,
            coverage=cov,
            candidates=candidates,
            selected_min=selected_min,
            selected_industrial=selected_ind,
            selected_industrial_with_mirror=selected_ind_with_mirror,
            selected_industrial_compose_light=selected_ind_compose_light,
            on_the_fly_policy=policy,
        )

        print("=== recomendación automática N (menos redundancia) ===")
        print(f"- Candidatas individuales analizadas: {n_total_individual}")
        print(
            f"- N_min recomendado (codo diversidad/coste): {n_opt} "
            f"(rango práctico: {n_low}-{n_high})"
        )
        print(
            f"- N_objetivo_industrial (robustez/comercial, acotado 20-80): {n_industrial}"
        )
        print(f"- Cobertura estimada de diversidad en N recomendado: {cov*100:.1f}%")
        print(
            "- Interpretación: por encima de ese N, suelen aparecer variantes "
            "más redundantes que útiles. El N_objetivo_industrial permite ampliar "
            "con control para ganar robustez."
        )
        print(
            f"- Composición mirror aplicada sobre seleccionadas (ratio={mirror_compose_ratio:.2f}): "
            f"{len(selected_ind_with_mirror) - len(selected_ind)} variantes extra"
        )
        print(
            f"- Compose light (máx 2 operaciones, ratio={compose_light_ratio:.2f}): "
            f"{len(selected_ind_compose_light)} variantes extra"
        )
        total_train_variants = len(selected_ind_with_mirror) + len(selected_ind_compose_light)
        print(
            "- TOTAL variantes seleccionadas para entrenamiento "
            f"(industrial + mirror compuesto + compose light): {total_train_variants}"
        )
        print("=== política recomendada augment on-the-fly (para train, aún no aplicada) ===")
        print("- Probabilidad de augment por muestra: 0.65")
        print("- Máximo operaciones por muestra: 2")
        print("- Mezcla de ops al augmentar: mirror=0.30 rotate=0.28 scale=0.16 shift=0.16 noise=0.10")
        print(
            "- Rangos sugeridos: "
            f"rotate=[{policy['ranges']['rotate_degrees'][0]:.3f},{policy['ranges']['rotate_degrees'][1]:.3f}] deg, "
            f"scale=[{policy['ranges']['scale_percentage'][0]:.3f},{policy['ranges']['scale_percentage'][1]:.3f}]%, "
            f"shift_dx=[{policy['ranges']['shift_dx'][0]:.4f},{policy['ranges']['shift_dx'][1]:.4f}], "
            f"shift_dy=[{policy['ranges']['shift_dy'][0]:.4f},{policy['ranges']['shift_dy'][1]:.4f}], "
            f"noise_sigma=[0..{policy['ranges']['noise_sigma_x'][1]:.4f}] aprox"
        )
        print(f"- Manifest de selección escrito en: {manifest_path}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida rangos de operaciones (mirror/rotate/scale/shift/noise) para un .npy "
            "sin salir de ventana o siendo compensable con shift."
        )
    )
    parser.add_argument("npy_path", type=str, help="Ruta al archivo .npy")
    parser.add_argument(
        "--config",
        type=str,
        default="validate_npy.json",
        help="Ruta al JSON de configuración por operación.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Perfil dentro del JSON (si existe 'profiles').",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifest.json",
        help="Ruta de salida del manifest JSON (se sobrescribe).",
    )
    parser.add_argument(
        "--mirror-compose-ratio",
        type=float,
        default=0.5,
        help=(
            "Proporción [0..1] de variantes industriales a las que se compone mirror "
            "(excepto si la variante ya es mirror)."
        ),
    )
    parser.add_argument(
        "--compose-light-ratio",
        type=float,
        default=0.35,
        help=(
            "Proporción [0..1] de variantes industriales para generar composición light "
            "(pipeline de máximo 2 operaciones)."
        ),
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help=(
            "Step global de fallback (si no hay step por operación en config). "
            "Ejemplos: 1 (enteros), 0.1, 0.01."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.npy_path):
        print(f"Archivo no encontrado: {args.npy_path}")
        raise SystemExit(1)
    cfg = _load_config(args.config)
    try:
        cfg_eff = _resolve_profile(cfg, args.profile)
    except ValueError as e:
        print(str(e))
        raise SystemExit(1)
    steps_cfg = cfg_eff.get("steps", {}) if isinstance(cfg_eff, dict) else {}
    rotate_cfg = cfg_eff.get("rotate", {}) if isinstance(cfg_eff, dict) else {}
    noise_cfg = cfg_eff.get("noise", {}) if isinstance(cfg_eff, dict) else {}

    global_step = float(args.step if args.step is not None else steps_cfg.get("global", 0.01))
    rotate_step = float(steps_cfg.get("rotate", global_step))
    scale_step = float(steps_cfg.get("scale", max(global_step * 100.0, 1e-9)))
    shift_step = float(steps_cfg.get("shift", global_step))
    noise_step = float(steps_cfg.get("noise", min(global_step, 0.001)))
    angle_min = float(rotate_cfg.get("min", -180.0))
    angle_max = float(rotate_cfg.get("max", 180.0))
    noise_sigma_cap = float(noise_cfg.get("sigma_cap", 0.05))

    for name, val in [
        ("global_step", global_step),
        ("rotate_step", rotate_step),
        ("scale_step", scale_step),
        ("shift_step", shift_step),
        ("noise_step", noise_step),
    ]:
        if val <= 0:
            print(f"{name} debe ser > 0")
            raise SystemExit(1)
    if args.mirror_compose_ratio < 0 or args.mirror_compose_ratio > 1:
        print("--mirror-compose-ratio debe estar en [0, 1]")
        raise SystemExit(1)
    if args.compose_light_ratio < 0 or args.compose_light_ratio > 1:
        print("--compose-light-ratio debe estar en [0, 1]")
        raise SystemExit(1)

    cfg_label = args.config if os.path.exists(args.config) else "(sin config, valores por defecto)"
    print(f"Config usada: {cfg_label}")
    print(f"Perfil: {args.profile}")
    print(
        f"steps => global={global_step}, rotate={rotate_step}, scale={scale_step}%, "
        f"shift={shift_step}, noise={noise_step}"
    )
    validate_npy(
        npy_path=args.npy_path,
        profile_name=args.profile,
        manifest_path=args.manifest,
        mirror_compose_ratio=args.mirror_compose_ratio,
        compose_light_ratio=args.compose_light_ratio,
        global_step=global_step,
        rotate_step=rotate_step,
        scale_step=scale_step,
        shift_step=shift_step,
        noise_step=noise_step,
        angle_min=angle_min,
        angle_max=angle_max,
        noise_sigma_cap=noise_sigma_cap,
    )


if __name__ == "__main__":
    main()
