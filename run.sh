#!/usr/bin/env bash
# PointMLP Tree Classification — Pipeline completo
# Uso: bash run.sh [--config configs/default.yaml]

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "=========================================="
echo " PointMLP Tree Classification Pipeline"
echo "=========================================="
echo "Config: $CONFIG"
echo ""

# Paso 1: Validar datos
echo "── Paso 1: Validando datos ──"
python scripts/validate_data.py --dataset real --config "$CONFIG"
echo ""
python scripts/validate_data.py --dataset synthetic --config "$CONFIG"
echo ""

# Paso 2: Preprocesar
echo "── Paso 2: Preprocesando datos ──"
python src/preprocess.py --dataset real --config "$CONFIG"
python src/preprocess.py --dataset synthetic --config "$CONFIG"
echo ""

# Paso 3: Entrenar
echo "── Paso 3: Entrenando modelos ──"
echo "Entrenando Model B (datos reales)..."
python src/train.py --dataset real --exp_name model_B --config "$CONFIG"
echo ""
echo "Entrenando Model A (datos sintéticos)..."
python src/train.py --dataset synthetic --exp_name model_A --config "$CONFIG"
echo ""

# Paso 4: Evaluar
echo "── Paso 4: Evaluando modelos ──"
python src/evaluate.py --exp_name model_A --config "$CONFIG"
python src/evaluate.py --exp_name model_B --config "$CONFIG"
echo ""

# Paso 5: Reporte comparativo
echo "── Paso 5: Generando reporte comparativo ──"
python src/report.py --models model_A model_B --config "$CONFIG"
echo ""

echo "=========================================="
echo " Pipeline completado"
echo "=========================================="
