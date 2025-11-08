#!/bin/bash
set -e

rm -rf .git
mkdir -p hf
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
cp -r ./app ./Dockerfile ./requirements.txt ./README.md hf/
cd hf
git config --global init.defaultBranch main
git init
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git remote add origin https://huggingface.co/spaces/MaxAtoms/Datathon-2025
git add .
git commit -m "Sync from GitHub"