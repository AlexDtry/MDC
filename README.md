# Droplets Apps

Portable Streamlit apps for droplet channel analysis and publication figure/table compilation.

## Included

- `apps/channel_analysis/droplets_channel_app.py`: YOLO droplet/channel analysis app.
- `apps/publication_compiler/compile_droplet_publication_app.py`: publication compiler app.
- `yolo_training/runs/segment/droplets_yolo11n_seg/weights/best.pt`: trained YOLO model required by the analysis app.
- Windows Conda installer and launchers.
- macOS local `.venv` installer and launchers.

For complete installation and troubleshooting instructions, read:

```text
README_PORTABLE.md
```

## Windows Quick Start

From PowerShell in this folder:

```powershell
.\install_dependencies_windows_conda.bat
.\apps\channel_analysis\lancer_channel_analysis_windows.bat
```

Publication compiler:

```powershell
.\apps\publication_compiler\lancer_publication_compiler_windows.bat
```

## macOS Quick Start

```bash
./install_dependencies_mac.command
apps/channel_analysis/lancer_channel_analysis.command
```

Publication compiler:

```bash
apps/publication_compiler/lancer_publication_compiler.command
```

