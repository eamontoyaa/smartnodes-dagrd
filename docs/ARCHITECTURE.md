# Arquitectura del MVP

Este MVP separa claramente tres responsabilidades:

```text
1. Datos
   data/raw/**/*.txt

2. Procesamiento reproducible
   src/sn_dagrd/*.py
   scripts/build_site.py

3. Publicación web estática
   site/index.html
   site/figures/**/*.png
   site/tables/**/*.csv
```

## Decisión principal

GitHub Pages solo sirve archivos estáticos. Por eso el procesamiento se ejecuta antes del despliegue mediante GitHub Actions. La página final no necesita backend.

## Flujo CI/CD

```text
push a main
    ↓
actions/checkout
    ↓
actions/setup-python
    ↓
instalación de dependencias
    ↓
pytest
    ↓
python scripts/build_site.py
    ↓
actions/upload-pages-artifact
    ↓
actions/deploy-pages
```

## Cuándo migrar a una arquitectura con backend

Este MVP sigue siendo adecuado si:

- Las lecturas se actualizan por commit.
- Las figuras pueden generarse de forma batch.
- No se requiere autenticación avanzada.
- No se requiere consulta dinámica por usuario.

Conviene migrar a una arquitectura con backend cuando se necesite:

- Ingesta remota en tiempo real.
- API para recibir datos desde nodos.
- Base de datos histórica.
- Control de usuarios.
- Dashboard interactivo con filtros dinámicos.
- Procesamiento bajo demanda.
