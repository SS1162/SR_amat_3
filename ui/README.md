# SPFI Studio — Dash UI Prototype

A UI-only prototype for the SPFI super-resolution model workspace, implemented with Dash and dash-bootstrap-components.

## Included screens

- **Training**
  - From Scratch
  - Source image upload
  - Optional mask upload
  - Output path / run configuration
  - Start / Stop controls
  - Preparation pipeline preview
  - Metric placeholders
- **Fine-Tuning**
  - Existing checkpoint selector
  - Dataset upload
  - UI scaffolding for future implementation
- **Inference**
  - Model selector
  - Single-image / batch mode
  - Upload + preview
  - Output preview placeholder
- **TensorBoard**
  - Dedicated monitoring panel ready for iframe/proxy integration

The demo callbacks are intentionally cosmetic and do **not** implement the real model computation.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:8050

## Design direction

- Dark research-tool aesthetic
- Deep navy background with cyan/violet accents
- Rounded glass-like panels
- Strong distinction between setup, monitoring, and output
- Responsive layout
- No JS framework beyond Dash
