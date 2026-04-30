#!/bin/zsh
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "Installation locale des dependances macOS."
echo "Ce script ne modifie pas Homebrew, Conda, ni Python systeme."

if [[ ! -d ".venv" ]]; then
  echo "Creation de l'environnement local .venv..."
  if [[ -x "/opt/homebrew/bin/python3.12" ]]; then
    /opt/homebrew/bin/python3.12 -m venv .venv
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  else
    echo "Erreur: python3 introuvable."
    exit 1
  fi
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Erreur: environnement .venv invalide."
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Dependances installees dans: $PROJECT_ROOT/.venv"
echo "Vous pouvez lancer les apps avec les fichiers .command dans apps/."
