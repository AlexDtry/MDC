@echo off
setlocal

set "APP_DIR=%~dp0"
pushd "%APP_DIR%\..\.." || exit /b 1

if "%DROPLETS_CONDA_ENV%"=="" (
  set "ENV_NAME=droplets_apps"
) else (
  set "ENV_NAME=%DROPLETS_CONDA_ENV%"
)

set "PORT=%STREAMLIT_PUBLICATION_PORT%"
if "%PORT%"=="" set "PORT=8503"
set "URL=http://localhost:%PORT%"

call install_dependencies_windows_conda.bat
if errorlevel 1 (
  popd
  exit /b 1
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
  echo Erreur: Conda est introuvable apres installation.
  popd
  exit /b 1
)

echo Lancement du compilateur publication sur %URL%...
start "" "%URL%"
call "%CONDA_CMD%" run -n "%ENV_NAME%" python -m streamlit run apps\publication_compiler\compile_droplet_publication_app.py --server.port "%PORT%" --server.headless true

popd
endlocal
exit /b 0
