from pathlib import Path
from typing import List, Dict, Any, Tuple

"""
Configuración específica para la parte de entrenamiento de modelos (training/).

- DATA_RESULT_ROOT: ruta a la carpeta data_result con los meta.json y user_X/poses.npy.
- EXPERIMENTS: catálogo de experimentos para train_model.py
"""


def suggest_split_ratios(n_examples: int) -> Tuple[float, float, float]:
    """
    Heurística única para preflight y train (si no pasas --train-ratio/--val-ratio).
    Con pocos ejemplos conviene más val/test; con muchos, más train.
    Devuelve (train, val, test) sumando 1.0.
    """
    n = int(max(0, n_examples))
    if n < 200:
        return 0.60, 0.20, 0.20
    if n < 2000:
        return 0.70, 0.15, 0.15
    if n < 20000:
        return 0.75, 0.125, 0.125
    return 0.80, 0.10, 0.10


# Carpeta data_result (salida de pose_extractor_clean: data_result/{cat}/{clip}/meta.json).
# También puedes usar GUADIA_DATA_RESULT_ROOT o --data-root en preflight/train.
DATA_RESULT_ROOT = Path("/home/debian/Proyectos/GuardIA/ResultadosExperimentos/data_result")

# Referencia para código que aún lee constantes (p. ej. CLI explícito o tests): banda "típica" 200≤N<2000.
# El entrenamiento por defecto usa suggest_split_ratios(N real) vía train_model_operations, no estos valores fijos.
SPLIT_RATIO_TRAIN, SPLIT_RATIO_VAL, SPLIT_RATIO_TEST = suggest_split_ratios(1500)

# Filtros de calidad por usuario/recorte (collect_examples, preflight, batch_build_manifest_cache).
# 0 = desactivado (no descartar por duración / frames / % válidos).
MIN_CLIP_SECONDS = 0.0
MIN_VALID_FRAMES = 0
MIN_VALID_PCT = 0.0
MAX_OCCLUSION_RATIO = 90.0

# Semilla (split, augment, PoseDataset).
SEED = 42

# Augment on-the-fly en train (probabilidad por muestra y ops consecutivas).
AUGMENT_PROB = 0.65
AUGMENT_MAX_OPS = 2
# Speed perturbation: factor < 1 acción más rápida; > 1 más lenta (interpolación temporal).
AUGMENT_SPEED_FACTOR_LO = 0.85
AUGMENT_SPEED_FACTOR_HI = 1.15

# Rejilla determinista alineada con validate_npy + cuánto mezclar con augment aleatorio en train.
MAX_DETERMINISTIC_VARIANTS = 64
TRAIN_DETERMINISTIC_PROB = 0.5

# Perfil dentro de operations_npy/validate_npy.json y lista selected_n_* en manifests por UID.
AUGMENT_PROFILE_DEFAULT = "industrial"
MANIFEST_VARIANT_SET_DEFAULT = "industrial"

# validate_npy.py / batch_build_manifest_cache (mismos defaults que el script CLI).
VALIDATE_NPY_MIRROR_COMPOSE_RATIO = 0.5
VALIDATE_NPY_COMPOSE_LIGHT_RATIO = 0.35

# Vistas extra por clip (train/val) desde manifests validate_npy: 0 = un ítem por .npy (comportamiento anterior).
EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT = 0

# Preflight: solo estimación de tiempos (--aug-variants-per-clip / --mirror-compose-ratio-estimate).
# - variantes por clip: típicamente |selected_n_objetivo_industrial| en el manifest (validate fija N en ~20–80
#   según candidatas y cobertura); 0 = no asumir expansión hasta que pases el flag.
# - mirror: alinea con validate_npy --mirror-compose-ratio (fracción de bases a las que se añade compose+mirror).
PREFLIGHT_AUG_VARIANTS_PER_CLIP = 0.0
PREFLIGHT_MIRROR_COMPOSE_RATIO_ESTIMATE = VALIDATE_NPY_MIRROR_COMPOSE_RATIO

# Variantes augmentadas por clip según categoría de acción (config_category_augmentation.json).
CATEGORY_AUGMENTATION_CONFIG_PATH = Path(__file__).parent / "config_category_augmentation.json"

# Clase positiva de robo (multiclass y referencia para preflight / balanceo de augment).
ROBBERY_CLASS = 6

# Preflight_train: objetivos asimétricos (recall robo vs falsos positivos).
PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS = 40
PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS = 15
PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO = 4.0
PREFLIGHT_ROBBERY_DOMINANCE_THRESHOLD = 0.25
PREFLIGHT_ROBBERY_RARE_THRESHOLD = 0.08

# Plan de entrenamiento (preflight_train_plan.py → training_plan.json).
TRAINING_PLAN_PATH = Path(__file__).parent / "training_plan.json"
# Eval binaria: comparar softmax-argmax vs umbral en P(robo) vs margen en logits crudos.
DEFAULT_BINARY_SOFTMAX_THRESHOLD = 0.8
DEFAULT_BINARY_LOGIT_MARGIN = 0.0


EXPERIMENTS: List[Dict[str, Any]] = [
    # ============================
    # TCNs: ligeras, medias y más profundas
    # ============================
    # Baselines rápidos (pocas épocas)
    {
        "arch": "tcn",
        "hidden_dim": 64,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 15,
        "pose_source": "filtered",  # "filtered" | "full"
        "done": False,
    },
    {
        "arch": "tcn",
        "hidden_dim": 64,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 25,
        "pose_source": "filtered",
        "done": False,
    },
    # TCN medias
    {
        "arch": "tcn",
        "hidden_dim": 128,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn",
        "hidden_dim": 128,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    # Más contexto temporal
    {
        "arch": "tcn",
        "hidden_dim": 128,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn",
        "hidden_dim": 256,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 60,
        "pose_source": "filtered",
        "done": False,
    },
    # Secuencias más cortas (acciones rápidas)
    {
        "arch": "tcn",
        "hidden_dim": 128,
        "dropout": 0.1,
        "seq_len": 32,
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 20,
        "pose_source": "filtered",
        "done": False,
    },
    # Secuencias más largas (acciones lentas)
    {
        "arch": "tcn",
        "hidden_dim": 256,
        "dropout": 0.2,
        "seq_len": 128,
        "batch_size": 32,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # TCN residual (res_tcn)
    # ============================
    {
        "arch": "res_tcn",
        "hidden_dim": 128,
        "num_blocks": 2,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # TCN dilatada (dilated_tcn)
    # ============================
    {
        "arch": "dilated_tcn",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "dilated_tcn",
        "hidden_dim": 128,
        "num_layers": 4,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "dilated_tcn",
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "res_tcn",
        "hidden_dim": 128,
        "num_blocks": 3,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "res_tcn",
        "hidden_dim": 256,
        "num_blocks": 3,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # LSTM / BiLSTM: simples y profundas
    # ============================
    # LSTM sencillos
    {
        "arch": "lstm",
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 25,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "lstm",
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    # BiLSTM más profundos
    {
        "arch": "lstm",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "lstm",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 60,
        "pose_source": "filtered",
        "done": False,
    },
    # Más contexto temporal
    {
        "arch": "lstm",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "lstm",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.4,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },
    # Secuencias cortas
    {
        "arch": "lstm",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_len": 32,
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    # Secuencias largas
    {
        "arch": "lstm",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 128,
        "batch_size": 32,
        "lr": 5e-4,
        "epochs": 60,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # Transformers temporales
    # ============================
    # Transformers ligeros
    {
        "arch": "transformer",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "transformer",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    # Más capas
    {
        "arch": "transformer",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    # Modelos más grandes
    {
        "arch": "transformer",
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "transformer",
        "d_model": 256,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 60,
        "pose_source": "filtered",
        "done": False,
    },
    # Secuencias cortas
    {
        "arch": "transformer",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "seq_len": 32,
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    # Secuencias largas
    {
        "arch": "transformer",
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 128,
        "batch_size": 32,
        "lr": 5e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # ST-GCN simplificado (stgcn)
    # ============================
    {
        "arch": "stgcn",
        "hidden_dim": 128,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "stgcn",
        "hidden_dim": 128,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "stgcn",
        "hidden_dim": 256,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # CNN 2D sobre mapas de pose (pose_cnn2d)
    # ============================
    {
        "arch": "pose_cnn2d",
        "hidden_dim": 64,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "pose_cnn2d",
        "hidden_dim": 64,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "pose_cnn2d",
        "hidden_dim": 128,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # Atención por articulación + temporal (joint_attn)
    # ============================
    {
        "arch": "joint_attn",
        "joint_d_model": 64,
        "temporal_d_model": 128,
        "joint_layers": 1,
        "temporal_layers": 2,
        "nhead": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "joint_attn",
        "joint_d_model": 64,
        "temporal_d_model": 128,
        "joint_layers": 2,
        "temporal_layers": 3,
        "nhead": 4,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "joint_attn",
        "joint_d_model": 64,
        "temporal_d_model": 256,
        "joint_layers": 1,
        "temporal_layers": 3,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },

    # ============================
    # Híbrido TCN + LSTM (tcn_lstm)
    # ============================
    {
        "arch": "tcn_lstm",
        "tcn_hidden_dim": 128,
        "tcn_layers": 2,
        "lstm_hidden_dim": 128,
        "lstm_layers": 1,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 30,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn_lstm",
        "tcn_hidden_dim": 128,
        "tcn_layers": 3,
        "lstm_hidden_dim": 128,
        "lstm_layers": 1,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn_lstm",
        "tcn_hidden_dim": 256,
        "tcn_layers": 2,
        "lstm_hidden_dim": 256,
        "lstm_layers": 1,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    # ============================
    # GRU / TCN+GRU
    # ============================
    {
        "arch": "gru",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 35,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "gru",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 7e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "gru",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 5e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },
    # ============================
    # GRU + atención temporal
    # ============================
    {
        "arch": "gru_attn",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 8e-4,
        "epochs": 40,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "gru_attn",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 55,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "gru_attn",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn_gru",
        "tcn_hidden_dim": 128,
        "tcn_layers": 2,
        "gru_hidden_dim": 128,
        "gru_layers": 1,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 45,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn_gru",
        "tcn_hidden_dim": 128,
        "tcn_layers": 3,
        "gru_hidden_dim": 128,
        "gru_layers": 1,
        "dropout": 0.25,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 60,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "tcn_gru",
        "tcn_hidden_dim": 256,
        "tcn_layers": 2,
        "gru_hidden_dim": 256,
        "gru_layers": 1,
        "dropout": 0.3,
        "seq_len": 64,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    # ============================
    # Conformer temporal ligero
    # ============================
    {
        "arch": "conformer_lite",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "conv_kernel": 7,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 8e-4,
        "epochs": 35,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "conformer_lite",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "conv_kernel": 7,
        "dropout": 0.15,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 6e-4,
        "epochs": 45,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "conformer_lite",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 384,
        "conv_kernel": 9,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "conformer_lite",
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 512,
        "conv_kernel": 9,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },
    # ============================
    # Nuevas arquitecturas (robustez comercial)
    # ============================
    # MS-TCN (multi-scale temporal conv)
    {
        "arch": "ms_tcn",
        "hidden_dim": 128,
        "num_blocks": 3,
        "dropout": 0.15,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 8e-4,
        "epochs": 35,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "ms_tcn",
        "hidden_dim": 128,
        "num_blocks": 4,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 6e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "ms_tcn",
        "hidden_dim": 256,
        "num_blocks": 4,
        "dropout": 0.25,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 4e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },
    # GAT espacial ligero + TCN temporal
    {
        "arch": "gat_tcn",
        "hidden_dim": 64,
        "tcn_hidden_dim": 128,
        "dropout": 0.15,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 8e-4,
        "epochs": 35,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "gat_tcn",
        "hidden_dim": 64,
        "tcn_hidden_dim": 128,
        "dropout": 0.2,
        "seq_len": 96,
        "batch_size": 64,
        "lr": 6e-4,
        "epochs": 50,
        "pose_source": "filtered",
        "done": False,
    },
    {
        "arch": "gat_tcn",
        "hidden_dim": 96,
        "tcn_hidden_dim": 256,
        "dropout": 0.25,
        "seq_len": 96,
        "batch_size": 32,
        "lr": 4e-4,
        "epochs": 70,
        "pose_source": "filtered",
        "done": False,
    },
]

