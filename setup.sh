#!/usr/bin/env bash
# Quick setup helper for Zero-Label Defect Hunter

set -e
echo "=== Zero-Label Defect Hunter setup ==="

if command -v conda &>/dev/null; then
  echo "Conda detected. You can run:"
  echo "  conda env create -f environment.yml"
  echo "  conda activate defect-hunter"
fi

python3 -m venv .venv
source .venv/bin/activate

echo "Installing CPU requirements (safest default)..."
pip install --upgrade pip
pip install -r requirements-cpu.txt

echo ""
echo "Training model on sample good images..."
python defect_hunter.py --data data --out models/defect_hunter_model.pkl

echo ""
echo "Done! Start the app with:"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo "Then open http://127.0.0.1:5000"
