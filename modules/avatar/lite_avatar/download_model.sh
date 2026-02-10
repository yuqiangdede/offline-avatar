#!/bin/bash
set -euo pipefail

echo "Downloading LiteAvatar model files..."
python download_model.py
echo "All model files downloaded successfully!"
