#!/usr/bin/env bash
# setup.sh – one-shot environment setup
set -e

echo "==========================================="
echo "  Fraud Detection System – Setup Script"
echo "==========================================="

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip --quiet
echo "[1/2] Installing Python packages …"
pip install -r requirements.txt --quiet

echo "[2/2] Setup complete!"
echo ""
echo "To run:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
