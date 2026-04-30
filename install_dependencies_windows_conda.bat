@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%" || exit /b 1

if "%DROPLETS_CONDA_ENV%"=="" (
  set "ENV_NAME=droplets_apps"
) else (
  set "ENV_NAME=%DROPLETS_CONDA_ENV%"
)

set "CONDA_CMD="
if not "%CONDA_EXE%"=="" if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"
if "%CONDA_CMD%"=="" (
  for /f "delims=" %%I in ('where conda 2^>nul') do (
    if "%CONDA_CMD%"=="" set "CONDA_CMD=%%I"
  )
)
if "%CONDA_CMD%"=="" if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if "%CONDA_CMD%"=="" if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\miniconda3\condabin\conda.bat"
if "%CONDA_CMD%"=="" if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if "%CONDA_CMD%"=="" if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%USERPROFILE%\anaconda3\condabin\conda.bat"
if "%CONDA_CMD%"=="" if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if "%CONDA_CMD%"=="" if exist "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
if "%CONDA_CMD%"=="" if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if "%CONDA_CMD%"=="" if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
if "%CONDA_CMD%"=="" if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\miniconda3\Scripts\conda.exe"
if "%CONDA_CMD%"=="" if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_CMD=%ProgramData%\miniconda3\condabin\conda.bat"
if "%CONDA_CMD%"=="" if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\anaconda3\Scripts\conda.exe"
if "%CONDA_CMD%"=="" if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_CMD=%ProgramData%\anaconda3\condabin\conda.bat"

if "%CONDA_CMD%"=="" (
  echo Erreur: Conda est introuvable.
  echo Installez Anaconda ou Miniconda, puis relancez ce fichier.
  echo Si Conda est deja installe, ouvrez "Anaconda Prompt" ou "Miniconda Prompt" dans ce dossier.
  popd
  exit /b 1
)

echo Conda detecte: %CONDA_CMD%
echo Environnement Conda: %ENV_NAME%

echo Verification des conditions d'utilisation Anaconda...
call "%CONDA_CMD%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
call "%CONDA_CMD%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul
call "%CONDA_CMD%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>nul

call "%CONDA_CMD%" run -n "%ENV_NAME%" python --version >nul 2>nul
if errorlevel 1 (
  echo Creation de l'environnement Conda %ENV_NAME%...
  call "%CONDA_CMD%" create -y -n "%ENV_NAME%" python=3.11 pip
  if errorlevel 1 (
    echo Erreur pendant la creation de l'environnement Conda.
    echo Lancez manuellement: conda create -y -n %ENV_NAME% python=3.11 pip
    popd
    exit /b 1
  )
) else (
  echo Environnement existant trouve. Verification/reparation des dependances...
)

echo Mise a jour de pip...
call "%CONDA_CMD%" run -n "%ENV_NAME%" python -m pip install --upgrade pip --timeout 120
if errorlevel 1 (
  echo Erreur pendant la mise a jour de pip.
  popd
  exit /b 1
)

echo Installation des dependances de requirements.txt...
call "%CONDA_CMD%" run -n "%ENV_NAME%" python -m pip install -r requirements.txt --timeout 120
if errorlevel 1 (
  echo Erreur pendant l'installation des dependances.
  echo Relancez ce fichier: l'installation peut reprendre si le reseau a coupe.
  popd
  exit /b 1
)

echo Verification des imports principaux...
call "%CONDA_CMD%" run -n "%ENV_NAME%" python -c "import streamlit, ultralytics, cv2, pandas, PIL, scipy, matplotlib; print('OK dependances')"
if errorlevel 1 (
  echo Erreur: les dependances semblent incompletes.
  popd
  exit /b 1
)

echo.
echo Installation terminee.
echo Environnement pret: %ENV_NAME%
popd
endlocal
exit /b 0
