#!/bin/bash
set -e

rm -rf .git
mkdir -p hf

# create README.md for Hugging Face Space
cat > README.md << EOF
---
title: Datathon 2025
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: unknown
app_port: 8050
---
EOF
cp -r ./app ./Dockerfile ./uv.lock ./pyproject.toml ./README.md hf/
cd hf

# global git setup
git config --global init.defaultBranch main
git lfs install

# local git setup
git init
git lfs track "*.parquet"
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git remote add origin https://huggingface.co/spaces/MaxAtoms/Datathon-2025
git add .
git commit -m "Sync from GitHub"
