import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Soportar ejecución como módulo (-m training.train_model) y como script (python training/train_model.py)
try:
    from .model_config import EXPERIMENTS, DATA_RESULT_ROOT  # type: ignore[attr-defined]
except ImportError:
    from model_config import EXPERIMENTS, DATA_RESULT_ROOT  # type: ignore[attr-defined]


# Semillas y splits
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # resto será test
MIN_SEQ_LEN = 4   # descartar secuencias demasiado cortas

# Modo debug: usar muy pocos datos y un experimento por arquitectura
DEBUG_MODE = True          # ponlo a True en local para pruebas rápidas
DEBUG_MAX_EXAMPLES = 5      # cuántos embeddings usar en total en debug

# Directorios locales para modelos y logs (dentro de training/)
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs" 
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class PoseExample:
    pose_path: Path
    label: int          # clase (original o binaria, según el modo)
    track_id: int
    clip_name: str
    category_str: str   # por si quieres inspeccionar


def get_data_result_root() -> Path:
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise RuntimeError(f"No se encontró la carpeta data_result en: {root}")
    return root


def collect_examples(pose_source: str = "filtered") -> List[PoseExample]:
    """
    Recorre data_result/{cat}/{clip_name}/ y construye ejemplos aplicando:
      1) Solo clips con un usuario, excepto categoría 6.
      2) En categoría 6, si hay varios usuarios, quedarse con el que tenga más frames.
      3) pose_source: "filtered" -> usa poses.npy, "full" -> usa poses_full.npy.
    """
    root = get_data_result_root()
    examples: List[PoseExample] = []

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_str = cat_dir.name
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            users = meta.get("users", [])
            if not users:
                continue

            # Filtrado por número de usuarios según tus reglas
            chosen_user = None
            if cat_str != "6":
                if len(users) != 1:
                    continue
                chosen_user = users[0]
            else:
                # cat == 6: elegir usuario con más frames totales
                chosen_user = max(users, key=lambda u: u.get("total_frames", 0))

            if not chosen_user:
                continue

            track_id = chosen_user.get("track_id")
            if track_id is None:
                continue

            user_dir = clip_dir / f"user_{track_id}"
            pose_filename = "poses.npy" if pose_source == "filtered" else "poses_full.npy"
            pose_path = user_dir / pose_filename
            if not pose_path.exists():
                continue

            try:
                poses = np.load(pose_path)
            except Exception:
                continue

            if poses.ndim != 3 or poses.shape[-1] != 2:
                continue
            if poses.shape[0] < MIN_SEQ_LEN:
                continue

            examples.append(
                PoseExample(
                    pose_path=pose_path,
                    label=int(meta.get("cat", cat_str)),
                    track_id=int(track_id),
                    clip_name=str(meta.get("clip_name", clip_dir.name)),
                    category_str=cat_str,
                )
            )

    if not examples:
        raise RuntimeError("No se encontraron ejemplos válidos en data_result.")
    return examples


def normalize_sequence(poses: np.ndarray) -> np.ndarray:
    """
    poses: [T, J, 2] con coordenadas normalizadas 0-1.
    Centra por la media de joints y escala por tamaño medio del cuerpo.
    """
    poses = poses.astype(np.float32)
    center = poses.mean(axis=1, keepdims=True)
    poses = poses - center
    scale = np.linalg.norm(poses, axis=-1).mean()
    if scale > 0:
        poses = poses / scale
    return poses


def add_velocity(poses: np.ndarray) -> np.ndarray:
    """
    poses: [T, J, 2] -> concatena velocidad: [T, J, 4] con (x,y,dx,dy).
    """
    vel = np.diff(poses, axis=0, prepend=poses[0:1])
    return np.concatenate([poses, vel], axis=-1)


def temporal_resize(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Redimensiona temporalmente una secuencia [T, ...] a [target_len, ...]
    con muestreo uniforme o padding por repetición.
    """
    t = seq.shape[0]
    if t == target_len:
        return seq
    if t > target_len:
        idx = np.linspace(0, t - 1, target_len).round().astype(int)
        return seq[idx]
    # t < target_len: padding repitiendo último frame
    pad_len = target_len - t
    pad = np.repeat(seq[-1:], pad_len, axis=0)
    return np.concatenate([seq, pad], axis=0)


class PoseDataset(Dataset):
    """
    Dataset que aplica:
      - normalización espacial
      - concatenación de velocidades
      - resize temporal a seq_len
      - flatten de joints a un vector de features por frame
    """

    def __init__(self, examples: List[PoseExample], label_to_idx: Dict[int, int], seq_len: int):
        self.examples = examples
        self.label_to_idx = label_to_idx
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ex = self.examples[idx]
        poses = np.load(ex.pose_path)  # [T, J, 2]
        poses = normalize_sequence(poses)
        poses = add_velocity(poses)  # [T, J, 4]
        poses = temporal_resize(poses, self.seq_len)  # [seq_len, J, 4]
        t, j, d = poses.shape
        poses = poses.reshape(t, j * d)  # [seq_len, F]
        x = torch.from_numpy(poses.astype(np.float32))  # [seq_len, F]
        y = self.label_to_idx[ex.label]
        return x, y


class PoseTCNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class PoseResTCNClassifier(nn.Module):
    """
    TCN residual más profunda:
      - Varios bloques Conv1d + ReLU + Dropout con conexiones residuales.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        h = self.in_proj(x)
        for block in self.blocks:
            residual = h
            h = block(h)
            h = h + residual
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class PoseDilatedTCNClassifier(nn.Module):
    """
    TCN con convoluciones dilatadas para captar dependencias largas en el tiempo.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            dilation *= 2
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        h = self.in_proj(x)
        for layer in self.layers:
            h = h + layer(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

class PoseLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        # Usamos el último estado temporal
        last = out[:, -1, :]  # [B, 2*hidden]
        return self.fc(last)


class PoseTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        b, t, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]
        cls_tokens = self.cls_token.expand(b, 1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+T, d_model]
        out = self.encoder(x)  # [B, 1+T, d_model]
        cls = out[:, 0, :]  # [B, d_model]
        return self.fc(cls)


class PoseSTGCNClassifier(nn.Module):
    """
    Versión muy simplificada de ST-GCN:
      - Reconstruye [B, T, J, F] a partir de [B, T, F*J]
      - Aplica una convolución de grafo fija sobre los joints
      - Luego una TCN 1D sobre el tiempo
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Asumimos 4 features por joint (x, y, dx, dy)
        if input_dim % 4 != 0:
            raise ValueError(f"PoseSTGCNClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4
        feat_per_joint = 4

        # Inicial: proyección por joint
        self.joint_mlp = nn.Linear(feat_per_joint, hidden_dim)

        # Adjacencia fija muy sencilla: cada joint conectado a sí mismo y vecinos inmediatos (cadena)
        A = torch.eye(self.num_joints)
        for j in range(self.num_joints - 1):
            A[j, j + 1] = 1.0
            A[j + 1, j] = 1.0
        # Normalización por grado
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A = A / deg
        self.register_buffer("A", A)  # [J, J]

        # TCN temporal después del grafo: trabajamos sobre canales=hidden_dim*num_joints
        tcn_input_dim = hidden_dim * self.num_joints
        self.tcn = nn.Sequential(
            nn.Conv1d(tcn_input_dim, tcn_input_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_input_dim, tcn_input_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(tcn_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] con F = J*4
        b, t, f = x.shape
        j = self.num_joints
        x = x.view(b, t, j, 4)  # [B, T, J, 4]

        # Proyección por joint
        x = self.joint_mlp(x)  # [B, T, J, H]

        # Grafo: para cada frame aplicamos A sobre la dimensión de joints
        # x_g[b, t, j, h] = sum_k A[j, k] * x[b, t, k, h]
        x = x.permute(0, 1, 3, 2)  # [B, T, H, J]
        x = torch.matmul(x, self.A.T)  # [B, T, H, J]
        x = x.permute(0, 1, 3, 2)  # [B, T, J, H]

        # Aplanar joints en canales y aplicar TCN temporal
        x = x.reshape(b, t, -1)  # [B, T, J*H]
        x = x.permute(0, 2, 1)   # [B, C=J*H, T]
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)  # [B, C]
        return self.fc(x)


class PoseCNN2DClassifier(nn.Module):
    """
    CNN 2D sobre "imágenes" de poses:
      - Reconstruye [B, T, J, 4] a partir de [B, T, F*J]
      - Forma un mapa [B, 4, T, J] (canales = x,y,dx,dy)
      - Aplica Conv2D + pooling y clasificador final
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(f"PoseCNN2DClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4

        self.cnn = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] con F = J*4
        b, t, f = x.shape
        j = self.num_joints
        x = x.view(b, t, j, 4)        # [B, T, J, 4]
        x = x.permute(0, 3, 1, 2)     # [B, 4, T, J]
        x = self.cnn(x)               # [B, C, T, J]
        x = self.pool(x).view(b, -1)  # [B, C]
        return self.fc(x)


class PoseJointAttnClassifier(nn.Module):
    """
    Modelo con atención por articulación + atención temporal:
      - Reconstruye [B, T, J, 4] a partir de [B, T, F*J]
      - Para cada frame, aplica un pequeño TransformerEncoder sobre los J joints (tokens = joints)
      - Obtiene un embedding por frame (media sobre joints)
      - Luego aplica un TransformerEncoder temporal sobre la secuencia de frames
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        joint_d_model: int = 64,
        temporal_d_model: int = 128,
        joint_layers: int = 1,
        temporal_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(f"PoseJointAttnClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4

        # Proyección por joint
        self.joint_proj = nn.Linear(4, joint_d_model)
        joint_encoder_layer = nn.TransformerEncoderLayer(
            d_model=joint_d_model,
            nhead=min(nhead, joint_d_model),
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.joint_encoder = nn.TransformerEncoder(joint_encoder_layer, num_layers=joint_layers)

        # Proyección a espacio temporal
        self.frame_proj = nn.Linear(joint_d_model, temporal_d_model)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_d_model,
            nhead=min(nhead, temporal_d_model),
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=temporal_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, temporal_d_model))
        self.fc = nn.Linear(temporal_d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] con F = J*4
        b, t, f = x.shape
        j = self.num_joints
        x = x.view(b, t, j, 4)  # [B, T, J, 4]

        # Atención por articulación (por frame)
        x = self.joint_proj(x)              # [B, T, J, Dj]
        x = x.view(b * t, j, -1)            # [B*T, J, Dj]
        x = self.joint_encoder(x)           # [B*T, J, Dj]
        x = x.mean(dim=1)                   # [B*T, Dj]  (media sobre joints)
        x = x.view(b, t, -1)                # [B, T, Dj]

        # Proyección a espacio temporal y Transformer temporal
        x = self.frame_proj(x)              # [B, T, Dt]
        cls_tokens = self.cls_token.expand(b, 1, -1)  # [B, 1, Dt]
        x = torch.cat([cls_tokens, x], dim=1)         # [B, 1+T, Dt]
        x = self.temporal_encoder(x)                  # [B, 1+T, Dt]
        cls = x[:, 0, :]                              # [B, Dt]
        return self.fc(cls)


class PoseTCNLSTMClassifier(nn.Module):
    """
    Híbrido TCN + BiLSTM:
      - TCN (Conv1d temporal) extrae features locales.
      - BiLSTM sobre la secuencia de features.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tcn_hidden_dim: int = 128,
        tcn_layers: int = 2,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, tcn_hidden_dim, kernel_size=1)

        tcn_blocks = []
        for _ in range(tcn_layers):
            tcn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.tcn_blocks = nn.ModuleList(tcn_blocks)

        self.lstm = nn.LSTM(
            input_size=tcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.permute(0, 2, 1)  # [B, F, T]
        h = self.in_proj(x)     # [B, C, T]
        for block in self.tcn_blocks:
            h = h + block(h)
        h = h.permute(0, 2, 1)  # [B, T, C]
        out, _ = self.lstm(h)   # [B, T, 2*H]
        last = out[:, -1, :]
        return self.fc(last)


def split_examples(examples: List[PoseExample]) -> Tuple[List[PoseExample], List[PoseExample], List[PoseExample]]:
    random.shuffle(examples)
    n = len(examples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]
    return train, val, test


def build_label_mapping(examples: List[PoseExample]) -> Dict[int, int]:
    labels = sorted({ex.label for ex in examples})
    return {lab: i for i, lab in enumerate(labels)}


def make_binary_examples(
    examples: List[PoseExample],
    positive_class: int = 6,
) -> List[PoseExample]:
    """
    Construye una lista de ejemplos binarios:
      - label = 1 si la clase original == positive_class
      - label = 0 en caso contrario
    """
    binary_examples: List[PoseExample] = []
    for ex in examples:
        bin_label = 1 if ex.label == positive_class else 0
        binary_examples.append(
            PoseExample(
                pose_path=ex.pose_path,
                label=bin_label,
                track_id=ex.track_id,
                clip_name=ex.clip_name,
                category_str=ex.category_str,
            )
        )
    return binary_examples


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate_with_metrics(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evalúa en un loader y devuelve:
      - pérdida media
      - accuracy
      - métricas detalladas: matriz de confusión, precision/recall/F1 por clase, macro/weighted F1 y top-3 accuracy.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    top3_correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        # Matriz de confusión
        for yt, yp in zip(y.tolist(), preds.tolist()):
            if 0 <= yt < num_classes and 0 <= yp < num_classes:
                conf_mat[yt][yp] += 1

        # Top-3 accuracy
        if logits.size(1) >= 3:
            top3 = logits.topk(3, dim=1).indices  # [B, 3]
            for yt, topk in zip(y.tolist(), top3.tolist()):
                if yt in topk:
                    top3_correct += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    # Métricas derivadas de la matriz de confusión
    per_class = {}
    supports = []
    f1s = []

    for c in range(num_classes):
        tp = conf_mat[c][c]
        fn = sum(conf_mat[c][j] for j in range(num_classes)) - tp
        fp = sum(conf_mat[i][c] for i in range(num_classes)) - tp
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        }

        supports.append(support)
        f1s.append(f1)

    total_support = sum(supports) if supports else 0
    if total_support > 0:
        macro_f1 = float(sum(f1s) / max(len(f1s), 1))
        weighted_f1 = float(
            sum(f * s for f, s in zip(f1s, supports)) / total_support
        )
    else:
        macro_f1 = 0.0
        weighted_f1 = 0.0

    top3_acc = top3_correct / max(total, 1) if total > 0 else 0.0

    metrics = {
        "confusion_matrix": conf_mat,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_acc": float(top3_acc),
    }
    return avg_loss, acc, metrics


def build_datasets_and_loaders(
    seq_len: int,
    batch_size: int,
    pose_source: str,
    num_workers: int = 4,
    task: str = "multiclass",
    positive_class: int = 6,
) -> Tuple[Dict[str, DataLoader], int, Dict[int, int]]:
    print(f"Recolectando ejemplos desde data_result... (pose_source='{pose_source}')")
    examples = collect_examples(pose_source=pose_source)
    print(f"Ejemplos totales (tras filtrado): {len(examples)}")

    if DEBUG_MODE:
        # Reducir drásticamente el número de ejemplos para pruebas locales rápidas
        examples = examples[:DEBUG_MAX_EXAMPLES]
        print(f"[DEBUG] Usando solo {len(examples)} ejemplos para train/val/test")

    # En modo binario reetiquetamos a 0/1 manteniendo el resto del flujo igual
    if task == "binary":
        print(f"[BINARIO] Usando clase positiva original: {positive_class}")
        examples = make_binary_examples(examples, positive_class=positive_class)

    train_ex, val_ex, test_ex = split_examples(examples)
    print(f"Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)}")

    label_to_idx = build_label_mapping(examples)
    num_classes = len(label_to_idx)
    print(f"Número de clases: {num_classes} | mapping: {label_to_idx}")

    train_ds = PoseDataset(train_ex, label_to_idx, seq_len)
    val_ds = PoseDataset(val_ex, label_to_idx, seq_len)
    test_ds = PoseDataset(test_ex, label_to_idx, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders, input_dim, label_to_idx


def build_model(arch: str, input_dim: int, num_classes: int, cfg: Dict[str, Any]) -> nn.Module:
    if arch == "tcn":
        return PoseTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "res_tcn":
        return PoseResTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_blocks=cfg.get("num_blocks", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "dilated_tcn":
        return PoseDilatedTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "stgcn":
        return PoseSTGCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "lstm":
        return PoseLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "transformer":
        return PoseTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "pose_cnn2d":
        return PoseCNN2DClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "joint_attn":
        return PoseJointAttnClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            joint_d_model=cfg.get("joint_d_model", 64),
            temporal_d_model=cfg.get("temporal_d_model", 128),
            joint_layers=cfg.get("joint_layers", 1),
            temporal_layers=cfg.get("temporal_layers", 2),
            nhead=cfg.get("nhead", 4),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "tcn_lstm":
        return PoseTCNLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
            tcn_layers=cfg.get("tcn_layers", 2),
            lstm_hidden_dim=cfg.get("lstm_hidden_dim", 128),
            lstm_layers=cfg.get("lstm_layers", 1),
            dropout=cfg.get("dropout", 0.1),
        )
    raise ValueError(f"Arquitectura desconocida: {arch}")


def run_experiment(
    exp_id: int,
    cfg: Dict[str, Any],
    device: torch.device,
    task: str = "multiclass",
    positive_class: int = 6,
    pose_source_override: str | None = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"Experimento {exp_id:02d} | config={cfg}")
    print("=" * 80)

    seq_len = cfg.get("seq_len", 64)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-3)
    epochs = cfg.get("epochs", 20)
    pose_source = pose_source_override or cfg.get("pose_source", "filtered")

    loaders, input_dim, label_to_idx = build_datasets_and_loaders(
        seq_len=seq_len,
        batch_size=batch_size,
        pose_source=pose_source,
        task=task,
        positive_class=positive_class,
    )
    num_classes = len(label_to_idx)

    model = build_model(cfg["arch"], input_dim, num_classes, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.0))

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        print(
            f"[Exp {exp_id:02d}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Exp {exp_id:02d}] Mejor val_acc: {best_val_acc:.4f}")

    test_loss, test_acc, test_metrics = evaluate_with_metrics(
        model,
        loaders["test"],
        criterion,
        device,
        num_classes=num_classes,
    )
    # Métricas agregadas en test (clips de un único usuario)
    macro_f1 = test_metrics["macro_f1"]
    weighted_f1 = test_metrics["weighted_f1"]

    # Si estamos en modo binario, extraemos también métricas específicas de la clase positiva (robos)
    f1_pos = None
    rec_pos = None
    prec_pos = None
    if task == "binary":
        # En binario, tras make_binary_examples, la clase 1 es la positiva
        # label_to_idx mapea label_original_binaria -> índice interno (normalmente {0:0, 1:1})
        pos_label = 1
        if pos_label in label_to_idx:
            pos_idx = label_to_idx[pos_label]
            pos_stats = test_metrics["per_class"].get(pos_idx, {})
            prec_pos = float(pos_stats.get("precision", 0.0))
            rec_pos = float(pos_stats.get("recall", 0.0))
            f1_pos = float(pos_stats.get("f1", 0.0))

    # Log a consola
    base_msg = (
        f"[Exp {exp_id:02d}] Test | "
        f"loss={test_loss:.4f} | acc={test_acc:.4f} | "
        f"macro_f1={macro_f1:.4f} | "
        f"weighted_f1={weighted_f1:.4f}"
    )
    if f1_pos is not None:
        base_msg += (
            f" | f1_pos={f1_pos:.4f} | "
            f"rec_pos={rec_pos:.4f} | prec_pos={prec_pos:.4f} "
            f"(clips de un único usuario, clase positiva={positive_class})"
        )
    print(base_msg)

    save_path = MODELS_DIR / f"modelo_{exp_id:02d}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "config": cfg,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "task": task,
        "positive_class": positive_class,
        "num_classes": num_classes,
        "metrics": {
            "best_val_acc": float(best_val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(macro_f1),
            "test_weighted_f1": float(weighted_f1),
            "test_f1_pos": float(f1_pos) if f1_pos is not None else None,
            "test_rec_pos": float(rec_pos) if rec_pos is not None else None,
            "test_prec_pos": float(prec_pos) if prec_pos is not None else None,
            "test_top3_acc": float(test_metrics["top3_acc"]),
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "test_per_class": test_metrics["per_class"],
            "history": history,
        },
    }

    torch.save(checkpoint, save_path)
    print(f"[Exp {exp_id:02d}] Modelo guardado en: {save_path}")

    return {
        "exp_id": exp_id,
        "config": cfg,
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_f1_pos": float(f1_pos) if f1_pos is not None else None,
        "test_rec_pos": float(rec_pos) if rec_pos is not None else None,
        "test_prec_pos": float(prec_pos) if prec_pos is not None else None,
        "save_path": str(save_path),
    }


def _select_debug_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    En modo debug seleccionamos:
      - El experimento más simple (menos epochs; si empatan, el primero) de cada arquitectura.
    """
    by_arch: Dict[str, Dict[str, Any]] = {}
    for cfg in experiments:
        arch = cfg.get("arch")
        if arch is None:
            continue
        if cfg.get("done", False):
            continue
        epochs = int(cfg.get("epochs", 0))
        if arch not in by_arch or epochs < int(by_arch[arch].get("epochs", 1e9)):
            by_arch[arch] = cfg
    selected = [by_arch[a] for a in sorted(by_arch.keys())]
    print(f"[DEBUG] Experimentos seleccionados (uno por arquitectura):")
    for cfg in selected:
        print(f"  - arch={cfg['arch']}, epochs={cfg.get('epochs')}, batch={cfg.get('batch_size')}, seq_len={cfg.get('seq_len')}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de acción sobre poses.")
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="Tipo de tarea: 'multiclass' (por defecto) o 'binary' (robo vs no-robo).",
    )
    parser.add_argument(
        "--positive-class",
        type=int,
        default=6,
        help="Etiqueta original considerada positiva en modo binario (por defecto 6 = robos).",
    )
    parser.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default=None,
        help="Sobrescribe pose_source de los experimentos: 'filtered' (poses.npy) o 'full' (poses_full.npy).",
    )
    return parser.parse_args()


def main():
    # Redirección de logs: terminal + fichero en training/logs/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_{timestamp}.log"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    if hasattr(s, "flush"):
                        s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    if hasattr(s, "flush"):
                        s.flush()
                except Exception:
                    pass

    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)

    try:
        args = parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {device}")
        print(f"Modelos se guardarán en: {MODELS_DIR}")
        print(f"Log de esta sesión: {log_path}")
        print(f"Tarea: {args.task} | positive_class={args.positive_class} | pose_source_override={args.pose_source}")

        results = []
        exps_iter = _select_debug_experiments(EXPERIMENTS) if DEBUG_MODE else EXPERIMENTS
        for i, cfg in enumerate(exps_iter, start=1):
            if (not DEBUG_MODE) and cfg.get("done", False):
                print(f"[Exp {i:02d}] Marcado como done=True, se omite.")
                continue
            res = run_experiment(
                i,
                cfg,
                device,
                task=args.task,
                positive_class=args.positive_class,
                pose_source_override=args.pose_source,
            )
            results.append(res)

        # Guardar resumen de todos los experimentos (junto al script de training)
        summary_path = BASE_DIR / "experiments_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Resumen de experimentos guardado en: {summary_path}")
    finally:
        # Restaurar stdout y cerrar log
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()

