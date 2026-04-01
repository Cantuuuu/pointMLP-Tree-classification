"""
Genera reporte HTML comparativo de los 3 flujos sobre FORinstance TLS.
Incluye: segmentación semántica, ITD, instancias (PQ/SQ/RQ) y visualizaciones.
"""
import json, base64
from pathlib import Path
from datetime import date

# ── Cargar datos ─────────────────────────────────────────────────────────────
results      = json.load(open("results/segmentation_forinstance/itd_results.json"))
inst_results = json.load(open("results/segmentation_forinstance/instance_results.json"))
rf_stats     = json.load(open("results/segmentation_forinstance/rf_train_stats.json"))

flows     = results["flows"]
inst_flows = inst_results["flows"]
A = flows["A"]; B = flows["B"]; C = flows["C"]
I0 = inst_flows["0"]; IA = inst_flows["A"]; IB = inst_flows["B"]; IC = inst_flows["C"]
itd_params = results["raster_itd_params"]
hag_thr    = results["flow_a_hag_threshold"]
ws_params  = inst_results["watershed_params"]
iou_thr    = inst_results["iou_threshold"]

FLOW_NAMES  = {"0": "Watershed Crudo", "A": "Geométrico (HAG)", "B": "PointNet++", "C": "Random Forest"}
FLOW_COLORS = {"0": "#607D8B", "A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}
SITES = ["CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"]

def agg(flow, key):
    return flow["segmentation_aggregate"][key]

def itd(flow, key):
    return flow["itd_aggregate"][key]

def iagg(iflow, key):
    return iflow["aggregate"][key]

def per_site_itd(flow_data, metric):
    site_data = {}
    for p in flow_data["per_plot"]:
        s = p["site"]
        if s not in site_data:
            site_data[s] = {"tp":0,"fp":0,"fn":0}
        itdp = p["itd"]
        site_data[s]["tp"] += itdp["tp"]
        site_data[s]["fp"] += itdp["fp"]
        site_data[s]["fn"] += itdp["fn"]
    result = {}
    for s, counts in site_data.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        result[s] = {"precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn,
                     "n_gt": tp+fn, "n_pred": tp+fp}
    return result

def per_site_inst(iflow_data):
    """Aggregate instance metrics (PQ/SQ/RQ) by site."""
    site_data = {}
    for p in iflow_data["per_plot"]:
        s = p["site"]
        if s not in site_data:
            site_data[s] = {"tp":0,"fp":0,"fn":0,"sq_sum":0.0,"miou_sum":0.0,"n_gt":0}
        inst = p["instances"]
        tp = inst["tp"]; fp = inst["fp"]; fn = inst["fn"]
        site_data[s]["tp"] += tp
        site_data[s]["fp"] += fp
        site_data[s]["fn"] += fn
        site_data[s]["sq_sum"]  += inst["sq"] * tp
        site_data[s]["miou_sum"] += inst["mean_inst_iou"] * inst["n_gt"]
        site_data[s]["n_gt"]    += inst["n_gt"]
    result = {}
    for s, d in site_data.items():
        tp = d["tp"]; fp = d["fp"]; fn = d["fn"]
        sq  = d["sq_sum"] / tp if tp > 0 else 0.0
        rq  = tp / (tp + 0.5*fp + 0.5*fn) if (tp+fp+fn) > 0 else 0.0
        pq  = sq * rq
        miou = d["miou_sum"] / d["n_gt"] if d["n_gt"] > 0 else 0.0
        result[s] = {"pq": pq, "sq": sq, "rq": rq, "mean_inst_iou": miou,
                     "tp": tp, "fp": fp, "fn": fn, "n_gt": d["n_gt"]}
    return result

# Pre-compute
site_itd  = {fid: per_site_itd(flows[fid], "f1")  for fid in ["A","B","C"]}
site_inst = {fid: per_site_inst(inst_flows[fid])   for fid in ["0","A","B","C"]}

def plot_rows():
    rows = []
    plots_a = {p["plot"]: p for p in A["per_plot"]}
    plots_b = {p["plot"]: p for p in B["per_plot"]}
    plots_c = {p["plot"]: p for p in C["per_plot"]}
    ip_a    = {p["plot"]: p for p in IA["per_plot"]}
    ip_b    = {p["plot"]: p for p in IB["per_plot"]}
    ip_c    = {p["plot"]: p for p in IC["per_plot"]}
    for p in A["per_plot"]:
        name = p["plot"]
        rows.append((p["site"], name, p["n_pts"], p["n_gt_trees"],
                     plots_a[name], plots_b.get(name,{}), plots_c.get(name,{}),
                     ip_a.get(name,{}), ip_b.get(name,{}), ip_c.get(name,{})))
    return rows

def img_b64(path: Path) -> str:
    """Embed PNG as base64 data URI (returns empty string if missing)."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"

VIS_DIR = Path("results/segmentation_forinstance/visualizations")

TODAY = date.today().isoformat()

# ── HTML ─────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comparación 3 Métodos — FORinstance TLS | {TODAY}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --a: #2196F3; --b: #FF5722; --c: #4CAF50;
    --bg: #f8f9fa; --card: #ffffff; --border: #dee2e6;
    --text: #212529; --muted: #6c757d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); line-height: 1.6; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 24px 16px; }}

  .header {{ background: linear-gradient(135deg,#1a237e,#283593);
             color:#fff; padding:40px 24px; border-radius:12px; margin-bottom:32px; }}
  .header h1 {{ font-size:2rem; font-weight:700; margin-bottom:8px; }}
  .header p  {{ opacity:.85; font-size:1rem; max-width:800px; }}
  .header .meta {{ margin-top:16px; display:flex; gap:24px; flex-wrap:wrap; }}
  .header .meta span {{ background:rgba(255,255,255,.15); padding:4px 12px;
                        border-radius:20px; font-size:.85rem; }}

  h2 {{ font-size:1.4rem; font-weight:600; margin:32px 0 16px;
       border-left:4px solid #1a237e; padding-left:12px; }}
  h3 {{ font-size:1.1rem; font-weight:600; margin:20px 0 10px; color:#1a237e; }}

  .card {{ background:var(--card); border:1px solid var(--border);
           border-radius:10px; padding:20px; margin-bottom:16px;
           box-shadow:0 1px 4px rgba(0,0,0,.06); }}

  .summary-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
  .flow-card {{ border-radius:10px; padding:20px; color:#fff; }}
  .flow-card.a {{ background:linear-gradient(135deg,#1976D2,#2196F3); }}
  .flow-card.b {{ background:linear-gradient(135deg,#E64A19,#FF5722); }}
  .flow-card.c {{ background:linear-gradient(135deg,#388E3C,#4CAF50); }}
  .flow-card h3 {{ color:#fff; margin:0 0 8px; font-size:1rem; }}
  .flow-card .big {{ font-size:2.2rem; font-weight:700; line-height:1; }}
  .flow-card .label {{ font-size:.75rem; opacity:.8; margin-bottom:10px; }}
  .flow-card .stats {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:5px; margin-top:10px; }}
  .flow-card .stat {{ background:rgba(0,0,0,.15); border-radius:6px; padding:5px 8px; }}
  .flow-card .stat .val {{ font-size:1rem; font-weight:700; }}
  .flow-card .stat .key {{ font-size:.65rem; opacity:.8; }}

  .tbl-wrap {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:.875rem; }}
  thead th {{ background:#1a237e; color:#fff; padding:10px 12px;
              text-align:left; font-weight:600; white-space:nowrap; }}
  tbody tr:nth-child(even) {{ background:#f8f9fa; }}
  tbody td {{ padding:8px 12px; border-bottom:1px solid var(--border); }}
  .best {{ font-weight:700; color:#1a237e; }}
  .site-badge {{ display:inline-block; padding:2px 8px; border-radius:10px;
                 font-size:.75rem; font-weight:600; color:#fff; }}
  .CULS    {{ background:#7B1FA2; }}
  .NIBIO   {{ background:#1565C0; }}
  .RMIT    {{ background:#B71C1C; }}
  .SCION   {{ background:#E65100; }}
  .TUWIEN  {{ background:#2E7D32; }}

  .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .chart-wrap {{ background:var(--card); border:1px solid var(--border);
                 border-radius:10px; padding:16px; }}
  .chart-wrap canvas {{ max-height:280px; }}

  .method-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
  .method-box {{ background:var(--card); border:1px solid var(--border);
                 border-radius:10px; padding:16px; }}
  .method-box .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
                        color:#fff; font-size:.8rem; font-weight:700; margin-bottom:10px; }}
  .method-box ul {{ padding-left:16px; font-size:.875rem; color:var(--muted); }}
  .method-box ul li {{ margin-bottom:4px; }}

  .highlight {{ background:#e8f5e9; border-left:4px solid #4CAF50;
                padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0;
                font-size:.9rem; }}
  .warn {{ background:#fff8e1; border-left:4px solid #FFC107;
           padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0;
           font-size:.9rem; }}
  .info {{ background:#e3f2fd; border-left:4px solid #2196F3;
           padding:12px 16px; border-radius:0 8px 8px 0; margin:12px 0;
           font-size:.9rem; }}

  /* Visualizations */
  .vis-section {{ margin-bottom:24px; }}
  .vis-row {{ display:flex; gap:12px; align-items:flex-start; margin-bottom:12px;
              flex-wrap:wrap; }}
  .vis-label {{ font-size:.8rem; font-weight:600; color:var(--muted);
                margin-bottom:4px; }}
  .vis-img {{ width:100%; border-radius:8px; border:1px solid var(--border);
              background:#111; }}
  .vis-plot-title {{ font-size:.9rem; font-weight:700; color:#1a237e;
                     margin:12px 0 6px; padding:6px 12px; background:#e8eaf6;
                     border-radius:6px; display:inline-block; }}
  .vis-col {{ flex:1; min-width:220px; }}

  @media(max-width:900px){{
    .summary-grid,.chart-grid,.method-grid{{ grid-template-columns:1fr; }}
    .vis-col {{ min-width:100%; }}
  }}
</style>
</head>
<body>
<div class="container">

<!-- HEADER -->
<div class="header">
  <h1>Segmentación de Instancias de Árboles — FORinstance TLS</h1>
  <p>Pipeline completo: segmentación semántica árbol/no-árbol → instancias por Watershed 2D →
     evaluación con métricas de Panoptic Quality (PQ/SQ/RQ). Dataset: FORinstance (LiDAR Terrestre TLS,
     escáneres Riegl VUX-1UAV y MiniVUX-1 UAV). Split oficial de test.</p>
  <div class="meta">
    <span>📅 {TODAY}</span>
    <span>🌲 Dataset: FORinstance TLS</span>
    <span>📊 11 plots · 5 sitios</span>
    <span>🌳 {sum(p['n_gt_trees'] for p in A['per_plot'])} árboles GT</span>
    <span>🔬 Watershed 2D (σ={ws_params['smooth_sigma']} · win={ws_params['local_max_window']} · cell={ws_params['cell_size']}m)</span>
    <span>📐 IoU threshold: {iou_thr}</span>
  </div>
</div>

<!-- PIPELINE SUMMARY -->
<h2>1. Pipeline de Segmentación de Instancias</h2>
<div class="card">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:.875rem">
    <div style="background:#e3f2fd;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:150px">
      <strong style="display:block;color:#1a237e">1. Nube de Puntos</strong>
      XYZ crudo TLS
    </div>
    <div style="font-size:1.5rem;color:#aaa">→</div>
    <div style="background:#e8f5e9;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:150px">
      <strong style="display:block;color:#2e7d32">2. Segmentación Semántica</strong>
      árbol / no-árbol por punto
    </div>
    <div style="font-size:1.5rem;color:#aaa">→</div>
    <div style="background:#fff8e1;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:150px">
      <strong style="display:block;color:#e65100">3. Raster Z-max</strong>
      resolución {ws_params['cell_size']} m/celda
    </div>
    <div style="font-size:1.5rem;color:#aaa">→</div>
    <div style="background:#fce4ec;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:150px">
      <strong style="display:block;color:#c62828">4. Watershed 2D</strong>
      copas → instancias
    </div>
    <div style="font-size:1.5rem;color:#aaa">→</div>
    <div style="background:#ede7f6;border-radius:8px;padding:10px 16px;text-align:center;flex:1;min-width:150px">
      <strong style="display:block;color:#4527a0">5. ID por Punto</strong>
      treeID predicho vs GT
    </div>
  </div>
  <div class="info" style="margin-top:16px">
    ℹ️ El mismo Watershed 2D se aplica a los 3 flujos: la única diferencia entre métodos
    es la <strong>segmentación semántica (paso 2)</strong>. Esto permite comparación justa de
    la capacidad de cada método para distinguir árbol/no-árbol, ya que el resto del pipeline es idéntico.
  </div>
</div>

<!-- RESUMEN EJECUTIVO -->
<h2>2. Resumen Ejecutivo — Segmentación de Instancias</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
  <div class="flow-card" style="background:linear-gradient(135deg,#455A64,#607D8B)">
    <h3>Flujo 0 — Watershed Crudo</h3>
    <div class="big">{iagg(I0,'pq'):.4f}</div>
    <div class="label">Panoptic Quality (PQ)</div>
    <div class="stats">
      <div class="stat"><div class="val">{iagg(I0,'sq'):.4f}</div><div class="key">SQ</div></div>
      <div class="stat"><div class="val">{iagg(I0,'rq'):.4f}</div><div class="key">RQ</div></div>
      <div class="stat"><div class="val">{iagg(I0,'mean_inst_iou'):.4f}</div><div class="key">mInstIoU</div></div>
      <div class="stat"><div class="val">—</div><div class="key">seg mIoU</div></div>
      <div class="stat"><div class="val">{iagg(I0,'tp')}</div><div class="key">TP inst</div></div>
      <div class="stat"><div class="val">{iagg(I0,'fp')}·{iagg(I0,'fn')}</div><div class="key">FP·FN</div></div>
    </div>
  </div>
  <div class="flow-card a">
    <h3>Flujo A — Geométrico (HAG)</h3>
    <div class="big">{iagg(IA,'pq'):.4f}</div>
    <div class="label">Panoptic Quality (PQ)</div>
    <div class="stats">
      <div class="stat"><div class="val">{iagg(IA,'sq'):.4f}</div><div class="key">SQ</div></div>
      <div class="stat"><div class="val">{iagg(IA,'rq'):.4f}</div><div class="key">RQ</div></div>
      <div class="stat"><div class="val">{iagg(IA,'mean_inst_iou'):.4f}</div><div class="key">mInstIoU</div></div>
      <div class="stat"><div class="val">{agg(A,'mean_iou'):.4f}</div><div class="key">seg mIoU</div></div>
      <div class="stat"><div class="val">{iagg(IA,'tp')}</div><div class="key">TP inst</div></div>
      <div class="stat"><div class="val">{iagg(IA,'fp')}·{iagg(IA,'fn')}</div><div class="key">FP·FN</div></div>
    </div>
  </div>
  <div class="flow-card b">
    <h3>Flujo B — PointNet++</h3>
    <div class="big">{iagg(IB,'pq'):.4f}</div>
    <div class="label">Panoptic Quality (PQ)</div>
    <div class="stats">
      <div class="stat"><div class="val">{iagg(IB,'sq'):.4f}</div><div class="key">SQ</div></div>
      <div class="stat"><div class="val">{iagg(IB,'rq'):.4f}</div><div class="key">RQ</div></div>
      <div class="stat"><div class="val">{iagg(IB,'mean_inst_iou'):.4f}</div><div class="key">mInstIoU</div></div>
      <div class="stat"><div class="val">{agg(B,'mean_iou'):.4f}</div><div class="key">seg mIoU</div></div>
      <div class="stat"><div class="val">{iagg(IB,'tp')}</div><div class="key">TP inst</div></div>
      <div class="stat"><div class="val">{iagg(IB,'fp')}·{iagg(IB,'fn')}</div><div class="key">FP·FN</div></div>
    </div>
  </div>
  <div class="flow-card c">
    <h3>Flujo C — Random Forest</h3>
    <div class="big">{iagg(IC,'pq'):.4f}</div>
    <div class="label">Panoptic Quality (PQ)</div>
    <div class="stats">
      <div class="stat"><div class="val">{iagg(IC,'sq'):.4f}</div><div class="key">SQ</div></div>
      <div class="stat"><div class="val">{iagg(IC,'rq'):.4f}</div><div class="key">RQ</div></div>
      <div class="stat"><div class="val">{iagg(IC,'mean_inst_iou'):.4f}</div><div class="key">mInstIoU</div></div>
      <div class="stat"><div class="val">{agg(C,'mean_iou'):.4f}</div><div class="key">seg mIoU</div></div>
      <div class="stat"><div class="val">{iagg(IC,'tp')}</div><div class="key">TP inst</div></div>
      <div class="stat"><div class="val">{iagg(IC,'fp')}·{iagg(IC,'fn')}</div><div class="key">FP·FN</div></div>
    </div>
  </div>
</div>

<!-- MÉTODOS -->
<h2>3. Descripción de Métodos</h2>
<div class="method-grid">
  <div class="method-box">
    <span class="badge" style="background:#2196F3">Flujo A — Baseline Geométrico</span>
    <p style="font-size:.875rem;margin-bottom:8px">Sin modelo entrenado. Solo geometría XYZ.</p>
    <ul>
      <li>HAG = Z − percentil 5 por celda 1 m</li>
      <li>Umbral: HAG &gt; {hag_thr} m → árbol</li>
      <li>Sin parámetros aprendidos</li>
      <li>seg mIoU = {agg(A,'mean_iou'):.4f}</li>
    </ul>
  </div>
  <div class="method-box">
    <span class="badge" style="background:#FF5722">Flujo B — PointNet++</span>
    <p style="font-size:.875rem;margin-bottom:8px">Red neuronal profunda para nubes de puntos.</p>
    <ul>
      <li>784,866 parámetros · PyTorch puro</li>
      <li>Entrenado 37 épocas · val IoU = 0.9746</li>
      <li>Bloques 5×5 m · multi-pass 4096 pts</li>
      <li>seg mIoU = {agg(B,'mean_iou'):.4f}</li>
    </ul>
  </div>
  <div class="method-box">
    <span class="badge" style="background:#4CAF50">Flujo C — Random Forest</span>
    <p style="font-size:.875rem;margin-bottom:8px">ML clásico con features geométricas.</p>
    <ul>
      <li>15 features: Z, kNN k=16/32, stats verticales</li>
      <li>200 árboles · max_depth=20 · OOB=0.985</li>
      <li>Sin GPU requerido</li>
      <li>seg mIoU = {agg(C,'mean_iou'):.4f}</li>
    </ul>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h3>Watershed 2D — Segmentación de Instancias</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:.875rem">
    <div>
      <strong>Analogía CHM-Watershed (ALS → TLS):</strong><br><br>
      En LiDAR aéreo (ALS/UAV-LS) se usa CHM-Watershed: raster de altura máxima
      de dosel → watershed por cuencas. Para TLS (Riegl VUX-1UAV / MiniVUX-1 UAV),
      se aplica el mismo principio desde la perspectiva terrestre:<br><br>
      <code style="background:#f1f3f4;padding:4px 8px;border-radius:4px;display:block;margin-top:4px">
        Raster Z-max → Suavizado Gaussiano → Máximos locales 2D → Watershed → ID por punto
      </code>
    </div>
    <div>
      <strong>Parámetros:</strong><br><br>
      <table style="width:100%;font-size:.8rem">
        <tr><td>Resolución raster</td><td><strong>{ws_params['cell_size']} m/celda</strong></td></tr>
        <tr><td>Suavizado Gaussiano σ</td><td><strong>{ws_params['smooth_sigma']} celdas</strong></td></tr>
        <tr><td>Ventana máximos locales</td><td><strong>{ws_params['local_max_window']} celdas
          ({ws_params['local_max_window']*ws_params['cell_size']} m)</strong></td></tr>
        <tr><td>IoU threshold matching</td><td><strong>{iou_thr}</strong></td></tr>
        <tr><td>Métrica principal</td><td><strong>Panoptic Quality (PQ = SQ × RQ)</strong></td></tr>
      </table>
    </div>
  </div>
</div>

<!-- GRÁFICAS -->
<h2>4. Comparación Visual</h2>
<div class="chart-grid">
  <div class="chart-wrap">
    <h3>Instancias — PQ · SQ · RQ · mInstIoU</h3>
    <canvas id="chartInst"></canvas>
  </div>
  <div class="chart-wrap">
    <h3>Segmentación — Métricas Agregadas</h3>
    <canvas id="chartSeg"></canvas>
  </div>
  <div class="chart-wrap">
    <h3>PQ por Sitio</h3>
    <canvas id="chartInstSite"></canvas>
  </div>
  <div class="chart-wrap">
    <h3>ITD F1 por Sitio</h3>
    <canvas id="chartSite"></canvas>
  </div>
</div>

<!-- TABLA COMPARATIVA PRINCIPAL -->
<h2>5. Tabla Comparativa Agregada</h2>
<div class="card">
<div class="tbl-wrap">
<table>
  <thead>
    <tr>
      <th rowspan="2">Flujo</th>
      <th colspan="3" style="text-align:center;background:#0d47a1">Segmentación Semántica</th>
      <th colspan="3" style="text-align:center;background:#b71c1c">ITD (Raster Z-max)</th>
      <th colspan="4" style="text-align:center;background:#1b5e20">Instancias (Watershed 2D)</th>
    </tr>
    <tr>
      <th style="background:#1565C0">seg mIoU</th><th style="background:#1565C0">tree IoU</th><th style="background:#1565C0">seg F1</th>
      <th style="background:#c62828">ITD Prec</th><th style="background:#c62828">ITD Recall</th><th style="background:#c62828">ITD F1</th>
      <th style="background:#2e7d32">PQ</th><th style="background:#2e7d32">SQ</th><th style="background:#2e7d32">RQ</th><th style="background:#2e7d32">mInstIoU</th>
    </tr>
  </thead>
  <tbody>
"""

best_pq    = max(iagg(I0,'pq'), iagg(IA,'pq'), iagg(IB,'pq'), iagg(IC,'pq'))
best_miou  = max(agg(A,'mean_iou'), agg(B,'mean_iou'), agg(C,'mean_iou'))
best_itdf1 = max(itd(A,'f1'), itd(B,'f1'), itd(C,'f1'))

def b(v, best, fmt=".4f"):
    return f'<span class="best">{v:{fmt}}</span>' if abs(v-best)<1e-5 else f'{v:{fmt}}'

# Flow 0 — no seg/ITD data
inst0 = I0["aggregate"]
html += f"""    <tr>
  <td><span class="badge" style="background:#607D8B">{FLOW_NAMES['0']}</span></td>
  <td colspan="3" style="color:#999;font-style:italic;text-align:center">sin segmentación semántica</td>
  <td colspan="3" style="color:#999;font-style:italic;text-align:center">sin ITD</td>
  <td>{b(inst0['pq'],best_pq)}</td><td>{inst0['sq']:.4f}</td><td>{inst0['rq']:.4f}</td><td>{inst0['mean_inst_iou']:.4f}</td>
</tr>\n"""

for fid, fdata, idata, css in [("A",A,IA,"a"),("B",B,IB,"b"),("C",C,IC,"c")]:
    s = fdata["segmentation_aggregate"]
    i = fdata["itd_aggregate"]
    inst = idata["aggregate"]
    html += f"""    <tr>
      <td><span class="badge" style="background:var(--{css})">{FLOW_NAMES[fid]}</span></td>
      <td>{b(s['mean_iou'],best_miou)}</td><td>{s['tree_iou']:.4f}</td><td>{s['f1']:.4f}</td>
      <td>{i['precision']:.4f}</td><td>{i['recall']:.4f}</td><td>{b(i['f1'],best_itdf1)}</td>
      <td>{b(inst['pq'],best_pq)}</td><td>{inst['sq']:.4f}</td><td>{inst['rq']:.4f}</td><td>{inst['mean_inst_iou']:.4f}</td>
    </tr>\n"""

html += """  </tbody>
</table>
</div>
</div>

<!-- RESULTADOS POR PLOT -->
<h2>6. Resultados por Plot</h2>
<div class="card">
<div class="tbl-wrap">
<table>
  <thead>
    <tr>
      <th>Sitio</th><th>Plot</th><th>Pts</th><th>GT</th>
      <th colspan="3" style="text-align:center">seg F1 (A/B/C)</th>
      <th colspan="3" style="text-align:center">ITD F1 (A/B/C)</th>
      <th colspan="3" style="text-align:center">PQ (A/B/C)</th>
      <th colspan="3" style="text-align:center">SQ (A/B/C)</th>
      <th colspan="3" style="text-align:center">n_pred inst (A/B/C)</th>
    </tr>
  </thead>
  <tbody>
"""

for site, name, n_pts, n_gt, pa, pb, pc, ia, ib, ic in plot_rows():
    sf_a = pa["segmentation"]["f1"] if "segmentation" in pa else 0
    sf_b = pb["segmentation"]["f1"] if "segmentation" in pb else 0
    sf_c = pc["segmentation"]["f1"] if "segmentation" in pc else 0
    if_a = pa["itd"]["f1"] if "itd" in pa else 0
    if_b = pb["itd"]["f1"] if "itd" in pb else 0
    if_c = pc["itd"]["f1"] if "itd" in pc else 0
    pq_a = ia["instances"]["pq"] if "instances" in ia else 0
    pq_b = ib["instances"]["pq"] if "instances" in ib else 0
    pq_c = ic["instances"]["pq"] if "instances" in ic else 0
    sq_a = ia["instances"]["sq"] if "instances" in ia else 0
    sq_b = ib["instances"]["sq"] if "instances" in ib else 0
    sq_c = ic["instances"]["sq"] if "instances" in ic else 0
    np_a = ia["instances"]["n_pred"] if "instances" in ia else 0
    np_b = ib["instances"]["n_pred"] if "instances" in ib else 0
    np_c = ic["instances"]["n_pred"] if "instances" in ic else 0
    best_sf = max(sf_a, sf_b, sf_c)
    best_if = max(if_a, if_b, if_c)
    best_pq_p = max(pq_a, pq_b, pq_c)
    def sfmt(v): return f'<span class="best">{v:.3f}</span>' if abs(v-best_sf)<1e-5 else f'{v:.3f}'
    def ifmt(v): return f'<span class="best">{v:.3f}</span>' if abs(v-best_if)<1e-5 else f'{v:.3f}'
    def pfmt(v): return f'<span class="best">{v:.3f}</span>' if abs(v-best_pq_p)<1e-5 else f'{v:.3f}'
    short = name.replace("_annotated","").replace("plot_","p")
    html += f"""    <tr>
      <td><span class="site-badge {site}">{site}</span></td>
      <td style="font-size:.8rem">{short}</td>
      <td style="font-size:.8rem">{n_pts:,}</td><td><strong>{n_gt}</strong></td>
      <td>{sfmt(sf_a)}</td><td>{sfmt(sf_b)}</td><td>{sfmt(sf_c)}</td>
      <td>{ifmt(if_a)}</td><td>{ifmt(if_b)}</td><td>{ifmt(if_c)}</td>
      <td>{pfmt(pq_a)}</td><td>{pfmt(pq_b)}</td><td>{pfmt(pq_c)}</td>
      <td>{sq_a:.3f}</td><td>{sq_b:.3f}</td><td>{sq_c:.3f}</td>
      <td>{np_a}</td><td>{np_b}</td><td>{np_c}</td>
    </tr>\n"""

html += """  </tbody>
</table>
</div>
</div>

<!-- INSTANCIAS POR SITIO -->
<h2>7. Instancias (PQ) por Sitio</h2>
<div class="card">
<div class="tbl-wrap">
<table>
  <thead>
    <tr>
      <th>Sitio</th><th>GT Trees</th>
      <th colspan="4">Flujo 0 — Watershed Crudo</th>
      <th colspan="4">Flujo A — Geométrico (HAG)</th>
      <th colspan="4">Flujo B — PointNet++</th>
      <th colspan="4">Flujo C — Random Forest</th>
    </tr>
    <tr>
      <th></th><th></th>
      <th>PQ</th><th>SQ</th><th>RQ</th><th>mIoU</th>
      <th>PQ</th><th>SQ</th><th>RQ</th><th>mIoU</th>
      <th>PQ</th><th>SQ</th><th>RQ</th><th>mIoU</th>
      <th>PQ</th><th>SQ</th><th>RQ</th><th>mIoU</th>
    </tr>
  </thead>
  <tbody>
"""

for site in SITES:
    d0 = site_inst["0"].get(site,{}); da = site_inst["A"].get(site,{})
    db = site_inst["B"].get(site,{}); dc = site_inst["C"].get(site,{})
    if not da: continue
    best_pq_s = max(d0.get("pq",0), da["pq"], db.get("pq",0), dc.get("pq",0))
    def bp(v): return f'<span class="best">{v:.3f}</span>' if abs(v-best_pq_s)<1e-5 else f'{v:.3f}'
    html += f"""    <tr>
      <td><span class="site-badge {site}">{site}</span></td>
      <td><strong>{da['n_gt']}</strong></td>
      <td>{bp(d0.get('pq',0))}</td><td>{d0.get('sq',0):.3f}</td><td>{d0.get('rq',0):.3f}</td><td>{d0.get('mean_inst_iou',0):.3f}</td>
      <td>{bp(da['pq'])}</td><td>{da['sq']:.3f}</td><td>{da['rq']:.3f}</td><td>{da['mean_inst_iou']:.3f}</td>
      <td>{bp(db.get('pq',0))}</td><td>{db.get('sq',0):.3f}</td><td>{db.get('rq',0):.3f}</td><td>{db.get('mean_inst_iou',0):.3f}</td>
      <td>{bp(dc.get('pq',0))}</td><td>{dc.get('sq',0):.3f}</td><td>{dc.get('rq',0):.3f}</td><td>{dc.get('mean_inst_iou',0):.3f}</td>
    </tr>\n"""

html += """  </tbody>
</table>
</div>
</div>
"""

# ── VISUALIZACIONES ────────────────────────────────────────────────────────────
html += "<h2>8. Visualizaciones — Antes / Después / Esperado</h2>\n"
html += """<div class="card">
<p style="font-size:.875rem;color:var(--muted);margin-bottom:16px">
  Cada figura muestra: <strong>ANTES</strong> (nube cruda coloreada por altura Z) ·
  <strong>DESPUÉS</strong> (instancias predichas, cada árbol = un color distinto) ·
  <strong>ESPERADO</strong> (GT treeID). Vista cenital XY (top-down, análoga a dron).
  Submuestra de 80,000 pts para visualización.
</p>\n"""

# Show sample plots for each flow
sample_plots = [
    ("CULS",  "plot_2_annotated"),
    ("SCION", "plot_61_annotated"),
    ("NIBIO", "plot_5_annotated"),
]
for site, plot_stem in sample_plots:
    html += f'<div class="vis-plot-title">📍 {site} / {plot_stem.replace("_annotated","")}</div>\n'
    html += '<div class="vis-row">\n'
    for fid, fname in [("0","Watershed Crudo"), ("A","Geométrico (HAG)"), ("B","PointNet++"), ("C","Random Forest")]:
        img_path = VIS_DIR / f"flow{fid}_{site}_{plot_stem}.png"
        src = img_b64(img_path)
        if src:
            html += f'''  <div class="vis-col">
    <div class="vis-label">Flujo {fid} — {fname}</div>
    <img class="vis-img" src="{src}" alt="Flow {fid} {site} {plot_stem}">
  </div>\n'''
        else:
            html += f'''  <div class="vis-col">
    <div class="vis-label">Flujo {fid} — {fname}</div>
    <div style="background:#eee;border-radius:8px;padding:20px;text-align:center;color:#999">
      Imagen no disponible
    </div>
  </div>\n'''
    html += '</div>\n'

html += '</div>\n'

# ── RF DETAILS ────────────────────────────────────────────────────────────────
html += f"""
<!-- RF DETAILS -->
<h2>9. Detalles del Random Forest</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
  <div class="card">
    <h3>Métricas de Entrenamiento</h3>
    <table style="width:100%;font-size:.875rem">
      <tr><td>OOB Accuracy</td><td><strong>{rf_stats['oob_score']:.4f}</strong></td></tr>
      <tr><td>Val mIoU</td><td><strong>{rf_stats['val_metrics']['mean_iou']:.4f}</strong></td></tr>
      <tr><td>Val tree IoU</td><td><strong>{rf_stats['val_metrics']['tree_iou']:.4f}</strong></td></tr>
      <tr><td>Test mIoU (bloques)</td><td><strong>{rf_stats['test_metrics']['mean_iou']:.4f}</strong></td></tr>
      <tr><td>Test tree IoU (bloques)</td><td><strong>{rf_stats['test_metrics']['tree_iou']:.4f}</strong></td></tr>
      <tr><td>Tiempo entrenamiento</td><td><strong>{rf_stats['timing']['training_s']:.0f}s ({rf_stats['timing']['training_s']/60:.1f} min)</strong></td></tr>
    </table>
  </div>
  <div class="card">
    <h3>Importancia de Features (Top 5)</h3>
    <table style="width:100%;font-size:.875rem">"""

top5 = sorted(rf_stats["feature_importances"].items(), key=lambda x:-x[1])[:5]
for feat, imp in top5:
    bar_w = int(imp * 400)
    html += f"""
      <tr>
        <td style="padding:4px 8px">{feat}</td>
        <td><div style="background:#4CAF50;width:{bar_w}px;height:16px;border-radius:3px;display:inline-block"></div>
            <span style="margin-left:6px;font-size:.8rem">{imp:.4f}</span></td>
      </tr>"""

html += f"""
    </table>
  </div>
</div>

<!-- ANÁLISIS -->
<h2>10. Análisis y Discusión</h2>
<div class="card">
  <h3>Segmentación: RF y HAG superan a PointNet++</h3>
  <p style="font-size:.875rem;margin-bottom:8px">
    Random Forest (mIoU={agg(C,'mean_iou'):.4f}) y Geométrico/HAG (mIoU={agg(A,'mean_iou'):.4f}) alcanzan
    prácticamente el mismo rendimiento en segmentación semántica, superando a PointNet++ (mIoU={agg(B,'mean_iou'):.4f}).
    La alta recall de HAG y RF (~0.99) indica detección casi perfecta de puntos arbóreos.
  </p>
  <div class="highlight">
    ✅ <strong>Segmentación:</strong> 15 features geométricas + RF logran la misma segmentación que el umbral
    HAG. La altura (Z, z_rank_pct, local_std) domina con ~60% de importancia. La separación árbol/no-árbol
    en TLS es intrínsecamente un problema de altura que no requiere aprendizaje profundo.
  </div>

  <h3 style="margin-top:16px">Instancias: baja PQ pero alta SQ — el cuello de botella es la separación</h3>
  <p style="font-size:.875rem;margin-bottom:8px">
    Los tres flujos obtienen PQ bajo ({iagg(IA,'pq'):.3f} / {iagg(IB,'pq'):.3f} / {iagg(IC,'pq'):.3f}),
    pero la SQ es moderada ({iagg(IA,'sq'):.3f} / {iagg(IB,'sq'):.3f} / {iagg(IC,'sq'):.3f}),
    lo que indica que <em>cuando una instancia se detecta correctamente, su segmentación es razonablemente precisa</em>.
    El problema principal es la RQ baja ({iagg(IA,'rq'):.3f} / {iagg(IB,'rq'):.3f} / {iagg(IC,'rq'):.3f}):
    over-segmentación (más instancias predichas que GT) y under-detection simultáneos.
  </p>
  <div class="warn">
    ⚠️ <strong>Limitación fundamental del Watershed 2D:</strong> En bosques TLS densos, dos árboles
    adyacentes pueden compartir un único máximo local en el raster Z-max cuando sus copas se solapan
    (→ under-detection) o un árbol con copa irregular puede generar múltiples máximos (→ over-segmentation).
    La solución requeriría Watershed 3D directo sobre la nube de puntos segmentada.
  </div>
  <div class="info">
    ℹ️ <strong>Flujo A > Flujo C > Flujo B en PQ.</strong> La mejor segmentación semántica (A y C vs B)
    se traduce directamente en mejor PQ: más puntos de árbol correctamente segmentados → raster Z-max
    más limpio → watershed más preciso. PointNet++ sufre en NIBIO (bosque boreal, diferente a los datos
    de entrenamiento), lo que degrada toda la cadena de instancias.
  </div>

  <h3 style="margin-top:16px">Variabilidad entre sitios</h3>
  <p style="font-size:.875rem">
    SCION (Nueva Zelanda, pinos de plantación) obtiene PQ más alto ({site_inst['A'].get('SCION',{}).get('pq',0):.3f} para Flujo A)
    gracias a la estructura regular de las copas. NIBIO (bosque boreal noruego) y TUWIEN (bosque mixto austríaco)
    presentan los valores más bajos, posiblemente por mayor solapamiento de copas y densidad variable.
    RMIT (Australia, eucaliptos) tiene la segmentación semántica más baja del Flujo A ({next((p['segmentation']['f1'] for p in A['per_plot'] if p['site']=='RMIT'), 0):.3f} seg F1),
    indicando que la vegetación australiana difiere estructuralmente del resto del dataset.
  </p>
</div>

<!-- PARÁMETROS Y REPRODUCIBILIDAD -->
<h2>11. Configuración y Reproducibilidad</h2>
<div class="card">
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;font-size:.875rem">
  <div>
    <strong>Preprocesamiento</strong><br>
    Block size: 5.0 m × 5.0 m<br>
    Stride: 2.5 m (50% overlap)<br>
    Pts/bloque: 4,096<br>
    Train: 2,902 bloques<br>
    Val: 511 bloques<br>
    Test: 1,484 bloques
  </div>
  <div>
    <strong>PointNet++ (Flujo B)</strong><br>
    SA layers: [1024, 256, 64]<br>
    Radii: [0.2, 0.5, 1.0] m<br>
    Best epoch: 37<br>
    Val IoU: 0.9746<br>
    Optimizer: AdamW lr=0.001<br>
    Loss: CE + Dice (0.5)
  </div>
  <div>
    <strong>Watershed Instancias</strong><br>
    Cell size: {ws_params['cell_size']} m<br>
    Gaussian σ: {ws_params['smooth_sigma']} celdas<br>
    Local max window: {ws_params['local_max_window']} celdas<br>
    IoU threshold: {iou_thr}<br>
    Métrica: PQ = SQ × RQ<br>
    Librería: skimage.watershed
  </div>
</div>
</div>

</div><!-- /container -->

<script>
"""

# Charts
pq_data   = [round(iagg(I0,'pq'),4),   round(iagg(IA,'pq'),4),   round(iagg(IB,'pq'),4),   round(iagg(IC,'pq'),4)]
sq_data   = [round(iagg(I0,'sq'),4),   round(iagg(IA,'sq'),4),   round(iagg(IB,'sq'),4),   round(iagg(IC,'sq'),4)]
rq_data   = [round(iagg(I0,'rq'),4),   round(iagg(IA,'rq'),4),   round(iagg(IB,'rq'),4),   round(iagg(IC,'rq'),4)]
miou_data = [round(iagg(I0,'mean_inst_iou'),4), round(iagg(IA,'mean_inst_iou'),4),
             round(iagg(IB,'mean_inst_iou'),4), round(iagg(IC,'mean_inst_iou'),4)]

site_pq_0 = [round(site_inst["0"].get(s,{}).get("pq",0),4) for s in SITES]
site_pq_a = [round(site_inst["A"].get(s,{}).get("pq",0),4) for s in SITES]
site_pq_b = [round(site_inst["B"].get(s,{}).get("pq",0),4) for s in SITES]
site_pq_c = [round(site_inst["C"].get(s,{}).get("pq",0),4) for s in SITES]
site_f1_a = [round(site_itd["A"].get(s,{}).get("f1",0),4) for s in SITES]
site_f1_b = [round(site_itd["B"].get(s,{}).get("f1",0),4) for s in SITES]
site_f1_c = [round(site_itd["C"].get(s,{}).get("f1",0),4) for s in SITES]

html += f"""
// Chart 1: Instance metrics (4 flows)
new Chart(document.getElementById('chartInst'), {{
  type:'bar',
  data:{{
    labels:['PQ','SQ','RQ','mInstIoU'],
    datasets:[
      {{ label:'Watershed Crudo',  backgroundColor:'rgba(96,125,139,.75)',
         data:[{pq_data[0]},{sq_data[0]},{rq_data[0]},{miou_data[0]}] }},
      {{ label:'Geométrico (HAG)', backgroundColor:'rgba(33,150,243,.75)',
         data:[{pq_data[1]},{sq_data[1]},{rq_data[1]},{miou_data[1]}] }},
      {{ label:'PointNet++',       backgroundColor:'rgba(255,87,34,.75)',
         data:[{pq_data[2]},{sq_data[2]},{rq_data[2]},{miou_data[2]}] }},
      {{ label:'Random Forest',    backgroundColor:'rgba(76,175,80,.75)',
         data:[{pq_data[3]},{sq_data[3]},{rq_data[3]},{miou_data[3]}] }},
    ]
  }},
  options:{{ responsive:true, scales:{{ y:{{ min:0, max:1 }} }},
             plugins:{{ legend:{{ position:'bottom' }} }} }}
}});

// Chart 2: Seg metrics (A/B/C only — Flow 0 has no semantic seg)
new Chart(document.getElementById('chartSeg'), {{
  type:'bar',
  data:{{
    labels:['mIoU','tree IoU','non-tree IoU','Precision','Recall','F1'],
    datasets:[
      {{ label:'Geométrico (HAG)', backgroundColor:'rgba(33,150,243,.75)',
         data:[{agg(A,'mean_iou'):.4f},{agg(A,'tree_iou'):.4f},{agg(A,'non_tree_iou'):.4f},{agg(A,'precision'):.4f},{agg(A,'recall'):.4f},{agg(A,'f1'):.4f}] }},
      {{ label:'PointNet++',       backgroundColor:'rgba(255,87,34,.75)',
         data:[{agg(B,'mean_iou'):.4f},{agg(B,'tree_iou'):.4f},{agg(B,'non_tree_iou'):.4f},{agg(B,'precision'):.4f},{agg(B,'recall'):.4f},{agg(B,'f1'):.4f}] }},
      {{ label:'Random Forest',    backgroundColor:'rgba(76,175,80,.75)',
         data:[{agg(C,'mean_iou'):.4f},{agg(C,'tree_iou'):.4f},{agg(C,'non_tree_iou'):.4f},{agg(C,'precision'):.4f},{agg(C,'recall'):.4f},{agg(C,'f1'):.4f}] }},
    ]
  }},
  options:{{ responsive:true, scales:{{ y:{{ min:0, max:1 }} }},
             plugins:{{ legend:{{ position:'bottom' }} }} }}
}});

// Chart 3: PQ by site (4 flows)
new Chart(document.getElementById('chartInstSite'), {{
  type:'bar',
  data:{{
    labels: {json.dumps(SITES)},
    datasets:[
      {{ label:'Watershed Crudo',  backgroundColor:'rgba(96,125,139,.75)', data:{site_pq_0} }},
      {{ label:'Geométrico (HAG)', backgroundColor:'rgba(33,150,243,.75)', data:{site_pq_a} }},
      {{ label:'PointNet++',       backgroundColor:'rgba(255,87,34,.75)',  data:{site_pq_b} }},
      {{ label:'Random Forest',    backgroundColor:'rgba(76,175,80,.75)',  data:{site_pq_c} }},
    ]
  }},
  options:{{ responsive:true, scales:{{ y:{{ min:0, max:1 }} }},
             plugins:{{ legend:{{ position:'bottom' }} }} }}
}});

// Chart 4: ITD F1 by site (A/B/C)
new Chart(document.getElementById('chartSite'), {{
  type:'bar',
  data:{{
    labels: {json.dumps(SITES)},
    datasets:[
      {{ label:'Geométrico (HAG)', backgroundColor:'rgba(33,150,243,.75)', data:{site_f1_a} }},
      {{ label:'PointNet++',       backgroundColor:'rgba(255,87,34,.75)',  data:{site_f1_b} }},
      {{ label:'Random Forest',    backgroundColor:'rgba(76,175,80,.75)',  data:{site_f1_c} }},
    ]
  }},
  options:{{ responsive:true, scales:{{ y:{{ min:0, max:1 }} }},
             plugins:{{ legend:{{ position:'bottom' }} }} }}
}});
</script>
</body>
</html>"""

out_path = Path("results/segmentation_forinstance/report_3methods_forinstance.html")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"Reporte guardado: {out_path}")
print(f"Tamaño: {out_path.stat().st_size/1024:.1f} KB")
