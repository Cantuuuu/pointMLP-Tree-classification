# PointMLP Tree Classification

Clasificación de especies de árboles usando **PointMLP** sobre nubes de puntos LiDAR.

## Objetivo

Comparar el rendimiento de dos modelos:

- **Model A**: entrenado con datos LiDAR **sintéticos**
- **Model B**: entrenado con datos LiDAR **reales** (IDTReeS/NEON)

Especies: ~4-6 géneros de árboles (Pinus, Quercus, etc.)

## Hardware requerido

| Componente | Mínimo |
|---|---|
| GPU | NVIDIA GTX 4060 (8 GB VRAM) |
| RAM | 32 GB |
| CPU | Intel i5-13420H o equivalente |
| CUDA | 12.x |

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar PyTorch con CUDA (ajustar URL según tu versión de CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar dependencias
pip install -r requirements.txt
```

## Preparar datos

### Datos reales (IDTReeS)

1. Descargar desde [IDTReeS](https://idtrees.org/) o [NEON](https://data.neonscience.org/)
2. Colocar archivos `.laz` en `data/raw/real/`
3. Incluir CSV de etiquetas (ver `data/raw/real/README.md`)

### Datos sintéticos

1. Obtener los `.laz` generados
2. Colocar en `data/raw/synthetic/`
3. Incluir CSV de etiquetas o campo embebido (ver `data/raw/synthetic/README.md`)

## Uso

### Pipeline completo

```bash
bash run.sh
```

### Paso a paso

```bash
# 1. Validar datos
python scripts/validate_data.py --dataset real
python scripts/validate_data.py --dataset synthetic

# 2. Preprocesar
python src/preprocess.py --dataset real
python src/preprocess.py --dataset synthetic

# 3. Entrenar
python src/train.py --dataset real --exp_name model_B
python src/train.py --dataset synthetic --exp_name model_A

# 4. Evaluar
python src/evaluate.py --exp_name model_A
python src/evaluate.py --exp_name model_B

# 5. Reporte comparativo
python src/report.py --models model_A model_B
```

## Estructura del proyecto

```
pointmlp-trees/
├── configs/default.yaml       # Hiperparámetros y configuración
├── data/
│   ├── raw/{real,synthetic}/  # Datos crudos (.laz)
│   └── processed/             # Datos preprocesados (.npy)
├── src/
│   ├── preprocess.py          # .laz → .npy
│   ├── dataset.py             # TreeDataset + DataLoader
│   ├── model.py               # Arquitectura PointMLP
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Métricas y confusion matrix
│   └── report.py              # Reporte comparativo
├── scripts/
│   └── validate_data.py       # Validación de datos crudos
├── results/{model_A,model_B}/ # Checkpoints y métricas
├── requirements.txt
└── run.sh                     # Pipeline completo
```

## Configuración

Editar `configs/default.yaml` para ajustar:
- Número de puntos por muestra (default: 1024)
- Batch size (default: 16, conservador para 8 GB VRAM)
- Epochs, learning rate, scheduler
- Mixed precision (habilitado por defecto)
