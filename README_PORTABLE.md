# Droplets apps portable

Ce dossier contient les deux applications Streamlit, leurs lanceurs, et le modele YOLO entraine.

## Ce qu'il faut garder

Ne renommez pas les dossiers internes apres extraction. L'app d'analyse cherche le modele ici:

```text
yolo_training\runs\segment\droplets_yolo11n_seg\weights\best.pt
```

## Windows: installation automatique avec Anaconda ou Miniconda

Prerequis: Anaconda ou Miniconda installe. Une connexion internet est necessaire la premiere fois.

Depuis PowerShell ou le Terminal Windows:

```powershell
cd "C:\chemin\vers\droplets_apps_portable"
.\install_dependencies_windows_conda.bat
```

Le script fait automatiquement:

- detection de Conda dans le PATH ou dans les emplacements Windows classiques;
- acceptation des Terms of Service Anaconda si la version de Conda le demande;
- creation de l'environnement `droplets_apps` avec Python 3.11;
- installation ou reparation des dependances;
- verification des imports principaux: Streamlit, Ultralytics, OpenCV, pandas, PIL, scipy, matplotlib.

Si l'environnement existe deja mais qu'une dependance manque, relancez simplement:

```powershell
.\install_dependencies_windows_conda.bat
```

## Windows: lancer les apps

App d'analyse YOLO:

```powershell
.\apps\channel_analysis\lancer_channel_analysis_windows.bat
```

Puis ouvrir:

```text
http://localhost:8501
```

Compilateur publication:

```powershell
.\apps\publication_compiler\lancer_publication_compiler_windows.bat
```

Puis ouvrir:

```text
http://localhost:8503
```

La fenetre PowerShell doit rester ouverte. Si elle est fermee, l'application s'arrete.

## macOS

Sur le Mac deja configure, lancez directement:

```bash
apps/channel_analysis/lancer_channel_analysis.command
apps/publication_compiler/lancer_publication_compiler.command
```

Si vous devez reinstaller les dependances locales sans toucher au systeme:

```bash
./install_dependencies_mac.command
```

Ce script macOS utilise seulement `.venv` dans ce dossier. Il ne modifie pas Homebrew, Conda, ni Python systeme.

## Diagnostic rapide Windows

Verifier que Conda marche:

```powershell
conda --version
```

Verifier que l'environnement est correct:

```powershell
conda run -n droplets_apps python -c "import streamlit, ultralytics, cv2, pandas; print('OK')"
```

Lancer l'app d'analyse sans launcher:

```powershell
conda run -n droplets_apps python -m streamlit run apps\channel_analysis\droplets_channel_app.py --server.port 8501 --server.headless true
```

Lancer l'app publication sans launcher:

```powershell
conda run -n droplets_apps python -m streamlit run apps\publication_compiler\compile_droplet_publication_app.py --server.port 8503 --server.headless true
```

## Problemes connus

Si Conda demande les Terms of Service, le script essaie de les accepter automatiquement. Si votre installation bloque encore, lancez:

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

Si `localhost:8501` est vide, regardez la fenetre PowerShell. Si la commande est revenue au prompt, l'app a plante et l'erreur est affichee dans le terminal.

Si le port 8501 est deja utilise:

```powershell
$env:STREAMLIT_PORT="8502"
.\apps\channel_analysis\lancer_channel_analysis_windows.bat
```

Si le port 8503 est deja utilise:

```powershell
$env:STREAMLIT_PUBLICATION_PORT="8504"
.\apps\publication_compiler\lancer_publication_compiler_windows.bat
```
