#!/bin/zsh
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$APP_DIR/../.." && pwd)"
PORT="${STREAMLIT_PUBLICATION_PORT:-8503}"
URL="http://localhost:${PORT}"

cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Creation de l'environnement Python..."
  if [[ -x "/opt/homebrew/bin/python3.12" ]]; then
    /opt/homebrew/bin/python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Environnement .venv invalide."
  exit 1
fi

if ! .venv/bin/python -m streamlit --version >/dev/null 2>&1; then
  echo "Installation des dependances..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Le compilateur publication semble deja lance sur ${URL}."
  open "$URL"
else
  echo "Lancement du compilateur publication sur ${URL}..."
  echo "La fenetre Terminal doit rester ouverte pour garder l'app active."
  (sleep 2 && open "$URL") &
  .venv/bin/python -m streamlit run apps/publication_compiler/compile_droplet_publication_app.py \
    --server.port "$PORT" \
    --server.headless true
fi
