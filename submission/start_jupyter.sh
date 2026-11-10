#!/bin/bash
# Organisers' guide: JupyterLab on port 9777, password/token zac2026; they open predict_notebook.ipynb and run all cells.
cd /code
jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --allow-root --no-browser
