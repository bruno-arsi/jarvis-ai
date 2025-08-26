#!/bin/bash
set -e

# Atualiza pacotes
sudo apt-get update

# Instala ffmpeg e libs de áudio
sudo apt-get install -y ffmpeg

# Instala dependências Python
pip install --upgrade pip
pip install -r requirements.txt