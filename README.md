# SmartNodes DAGRD

MVP estático para publicar resultados de lecturas de inclinómetros remotos SmartNode/DAGRD mediante GitHub Pages.

El repositorio está pensado para este flujo:

```text
lecturas JSONL en data/raw/
        ↓
scripts/build_site.py
        ↓
lectura de quaternions y ángulos Euler
        ↓
corrección de deriva por segmentos usando mediana móvil
        ↓
cálculo de desplazamientos acumulados
        ↓
figuras PNG + tablas CSV en site/
        ↓
GitHub Pages
```

## Estructura del repositorio

```text
smartnodes-dagrd/
├── data/
│   └── raw/
│       ├── la_palmera/
│       │   └── raw_InclinometroLaPalmera.txt
│       └── santa_rita_1/
│           └── raw_InclinometroSantaRita1.txt
├── notebooks/
│   ├── inc_smartnode_la_palmera_drift_corr.ipynb
│   └── inc_smartnode_sta_rita_1_drift_corr.ipynb
├── scripts/
│   └── build_site.py
├── site/
│   ├── index.html
│   ├── assets/
│   ├── figures/
│   └── tables/
├── src/
│   └── sn_dagrd/
├── tests/
└── .github/
    └── workflows/
        └── deploy-pages.yml
```

## Instalación local

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generar el sitio localmente

```bash
python scripts/build_site.py
```

Luego abre:

```text
site/index.html
```

El script genera:

```text
site/figures/<estacion>/*.png
site/tables/<estacion>/*.csv
site/tables/summary.json
site/index.html
```

## Ejecutar prueba rápida

```bash
pytest -q
```

## Actualizar las lecturas

Para actualizar una estación existente, reemplaza o modifica el archivo correspondiente:

```text
data/raw/santa_rita_1/raw_InclinometroSantaRita1.txt
data/raw/la_palmera/raw_InclinometroLaPalmera.txt
```

Después:

```bash
git add data/raw scripts src README.md

git commit -m "Update inclinometer readings"

git push origin main
```

Cada push a `main` ejecuta el workflow `.github/workflows/deploy-pages.yml`, que instala dependencias, corre las pruebas, reconstruye `site/` y publica el resultado en GitHub Pages.

## Activar GitHub Pages

En GitHub:

```text
Repository → Settings → Pages → Build and deployment → Source → GitHub Actions
```

Después de hacer push a `main`, entra a:

```text
Repository → Actions → Build and deploy SmartNodes MVP
```

Cuando el job termine, GitHub mostrará la URL pública del sitio en el ambiente `github-pages`.

## Configuración de estaciones

Las estaciones están configuradas dentro de `scripts/build_site.py` en la lista `STATIONS`.

Ejemplo abreviado:

```python
StationConfig(
    slug="la_palmera",
    name="La Palmera",
    code="VLP",
    station_id="SmartNode-LaPalmera",
    raw_path=ROOT / "data" / "raw" / "la_palmera" / "raw_InclinometroLaPalmera.txt",
    valid_sensors={"1a": 1, "2a": 0, "3a": 1},
    depths_m=np.linspace(-0.25, 14.825, num=15),
    azimuth_deg=295.0,
    start_date="2026-04-06",
    sample_freq="6h",
    sensor_to_plot="3a",
)
```

Para agregar otra estación:

1. Crea una carpeta en `data/raw/<nueva_estacion>/`.
2. Agrega el archivo JSONL de lecturas.
3. Agrega una nueva entrada `StationConfig`.
4. Ejecuta `python scripts/build_site.py`.
5. Revisa `site/index.html`.
6. Haz commit y push.

## Salidas principales del sitio

Por estación, se publican estas figuras:

- Perfil corregido más reciente.
- Perfiles acumulados corregidos.
- Pitch original vs corregido.
- Roll original vs corregido.
- Evolución temporal de un sensor representativo.
- Disponibilidad de quaternions por sensor.
- RSSI.
- Perfiles acumulados sin corrección.

También se publican tablas CSV con disponibilidad y último perfil de desplazamiento.

## Nota sobre seguridad

Este MVP no usa secretos, tokens, base de datos ni backend. No subas archivos `.env`, credenciales, passwords, tokens de GitHub, URLs privadas con contraseña ni llaves de servicios externos.
