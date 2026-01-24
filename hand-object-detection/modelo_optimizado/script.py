import torch
import time
from torch.quantization import quantize_dynamic

# -------------------------
# CONFIGURACIÓN
# -------------------------
MODEL_PATH = "modelo.pth"
N_RUNS = 100
WARMUP = 10

torch.set_num_threads(torch.get_num_threads())
device = torch.device("cpu")

# -------------------------
# CARGAR MODELO ORIGINAL
# -------------------------
print("Cargando modelo original...")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# ⚠️ Si usas state_dict, sería:
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# -------------------------
# INPUT DE PRUEBA (AJUSTA ESTO)
# -------------------------
dummy_input = torch.randn(1, 512)

# -------------------------
# BENCHMARK FUNCIÓN
# -------------------------
def benchmark(model, name):
    print(f"\nBenchmark: {name}")

    # warm-up
    for _ in range(WARMUP):
        _ = model(dummy_input)

    t0 = time.time()
    for _ in range(N_RUNS):
        _ = model(dummy_input)
    t1 = time.time()

    avg_time = (t1 - t0) / N_RUNS
    print(f"Tiempo medio inferencia: {avg_time:.6f} s")
    return avg_time

# -------------------------
# BENCHMARK ORIGINAL
# -------------------------
t_original = benchmark(model, "Modelo ORIGINAL")

# -------------------------
# CUANTIZACIÓN INT8 (CPU)
# -------------------------
print("\nCuantizando modelo a INT8 (CPU)...")
t0 = time.time()

model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

t1 = time.time()
print(f"Tiempo de cuantización: {t1 - t0:.2f} s")

# -------------------------
# GUARDAR MODELO OPTIMIZADO
# -------------------------
torch.save(model_int8.state_dict(), "modelo_int8_cpu.pth")
print("Modelo INT8 guardado")

# -------------------------
# BENCHMARK OPTIMIZADO
# -------------------------
t_int8 = benchmark(model_int8, "Modelo INT8")

# -------------------------
# COMPARACIÓN
# -------------------------
speedup = t_original / t_int8

print("\n================ RESULTADOS ================")
print(f"Original : {t_original:.6f} s")
print(f"INT8     : {t_int8:.6f} s")
print(f"Speedup  : {speedup:.2f}x")
print("============================================")
