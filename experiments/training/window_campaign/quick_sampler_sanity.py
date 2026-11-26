#!/usr/bin/env python3
"""En ~2s: valida fórmula del sampler (bug rec_robo=0 en corpus grande)."""
from __future__ import annotations

from window_index import WindowExample
from window_sampler import HierarchicalClipCapSampler


def _fake_windows(n_pos: int, n_neg: int) -> list[WindowExample]:
    out: list[WindowExample] = []
    for i in range(n_pos):
        out.append(
            WindowExample(
                pose_path=__import__("pathlib").Path(f"/tmp/p{i}.npy"),
                valid_mask_path=None,
                label=1,
                clip_name=f"p{i}",
                category_str="6",
                track_id=0,
                action_label=None,
                start=0,
                end=96,
                window_id=f"p{i}::w0000",
                clip_uid=f"6::p{i}::u0",
                label_state="pos_core",
                train_label=1,
                sample_weight=2.0,
            )
        )
    for i in range(n_neg):
        out.append(
            WindowExample(
                pose_path=__import__("pathlib").Path(f"/tmp/n{i}.npy"),
                valid_mask_path=None,
                label=0,
                clip_name=f"n{i}",
                category_str="3",
                track_id=0,
                action_label=None,
                start=0,
                end=96,
                window_id=f"n{i}::w0000",
                clip_uid=f"3::n{i}::u0",
                label_state="neg",
                train_label=0,
                sample_weight=1.0,
            )
        )
    return out


def _frac_pos(sampler: HierarchicalClipCapSampler, windows: list[WindowExample]) -> float:
    n1 = n = 0
    for idx in sampler:
        n += 1
        if int(windows[idx].train_label) == 1:
            n1 += 1
    return n1 / max(1, n)


def main() -> int:
    prod_pos, prod_neg, prod_len = 3135, 17044, 20444
    batch, fixed = 32, 3200
    old_n = max(batch * 100, prod_len)
    old_target = int(old_n * 0.35)
    print(
        f"[MATH] Corpus real ~len={prod_len} | ANTIGUO num_samples={old_n} "
        f"→ pide {old_target} positivos/época pero pool≈{prod_pos} "
        f"→ relleno masivo con neg (rec_robo=0)."
    )
    print(
        f"[MATH] FIX config samples_per_epoch={fixed} share=0.4 "
        f"→ target≈{int(fixed * 0.4)} (acotado al pool)."
    )

    # Mini simulación proporcional (1 clip / ventana para iterar rápido)
    mini = _fake_windows(200, 800)
    good = HierarchicalClipCapSampler(
        mini,
        num_samples=min(fixed, len(mini)),
        positive_share=0.4,
        max_windows_per_clip=4,
    )
    frac = _frac_pos(good, mini)
    print(f"[SAMPLER-FIX] {good.stats_line()} | frac_pos_emitido={100*frac:.1f}%")
    ok = frac >= 0.28
    print("[SAMPLER] OK" if ok else "[SAMPLER] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())