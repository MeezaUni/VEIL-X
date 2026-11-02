@echo off
rem ==========================================================================
rem  VEIL-X one-click launcher (Windows 10/11)
rem
rem  First run: installs Python 3.12 and all dependencies into .venv-app,
rem  downloads the AdaFace / ArcFace weights from Hugging Face, then starts
rem  the app. Later runs start the app directly.
rem
rem  Needs: internet on the first run, about 6 GB of free disk space, and
rem  the private runtime pack VEIL-X_runtime_pack.zip placed next to this
rem  file (ask the project owner). An NVIDIA GPU is optional.
rem
rem  Options are passed on to the app, for example:
rem    run_veilx.bat --port 7861
rem    run_veilx.bat --device cpu
rem ==========================================================================
setlocal EnableExtensions
title VEIL-X launcher
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HERE=%~dp0"
set "VENV=%HERE%.venv-app"
set "PY=%VENV%\Scripts\python.exe"
if not defined VEILX_DATA_ROOT set "VEILX_DATA_ROOT=%HERE%veilx_data"

echo.
echo  ==============================================================
echo   VEIL-X: AI face verification with quality assessment
echo           and intelligent abstention
echo   Local academic demonstration - not a certified identity system
echo  ==============================================================
echo.

rem ---- 1. runtime pack: frozen files that are not on GitHub ----
if exist "%VEILX_DATA_ROOT%\reports\phase3_frozen.json" goto have_pack
if not exist "%HERE%VEIL-X_runtime_pack.zip" goto no_pack
echo [1/4] Extracting VEIL-X_runtime_pack.zip ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force -LiteralPath '%HERE%VEIL-X_runtime_pack.zip' -DestinationPath '%HERE%'"
if errorlevel 1 goto fail
if not exist "%VEILX_DATA_ROOT%\reports\phase3_frozen.json" goto no_pack
:have_pack
echo [1/4] Runtime pack found: %VEILX_DATA_ROOT%

rem ---- 2. uv: installs Python 3.12 automatically ----
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo.
echo  VEIL-X uses "uv" from astral.sh to install Python 3.12 and the
echo  libraries automatically, without changing your system Python.
choice /C YN /M " Install uv now"
if errorlevel 2 goto no_uv
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto no_uv
:have_uv
echo [2/4] uv found.

rem ---- 3. Python environment, first run only ----
if exist "%VENV%\.veilx_ready" goto have_env
echo [3/4] First-time setup: Python 3.12 + libraries. This takes 5-15 minutes.
uv venv --python 3.12 "%VENV%"
if errorlevel 1 goto fail
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
nvidia-smi >nul 2>nul
if not errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
echo        PyTorch build: %TORCH_INDEX%
uv pip install --python "%PY%" torch --index-url %TORCH_INDEX%
if errorlevel 1 goto fail
uv pip install --python "%PY%" -e ".[gpu,app]"
if errorlevel 1 goto fail
echo ok> "%VENV%\.veilx_ready"
:have_env
echo [3/4] Python environment ready.

rem ---- 4. face-recognition weights, public, from Hugging Face ----
if exist "%VEILX_DATA_ROOT%\models\minchul__cvlface_adaface_ir101_webface4m\f2b38d9e24bfe301490d8dd081d8924b102333dd\model.safetensors" if exist "%VEILX_DATA_ROOT%\models\minchul__cvlface_arcface_ir101_webface4m\5846a5e96fbe1094952a414c78ce5b057c92d013\model.safetensors" goto have_models
echo [4/4] Downloading the AdaFace and ArcFace weights, about 520 MB ...
"%PY%" scripts\fetch_models.py
if errorlevel 1 goto fail
:have_models
echo [4/4] Model weights ready.

echo.
echo  Starting VEIL-X. Your browser opens automatically - if not, open the
echo  address printed below, normally http://127.0.0.1:7860
echo  Keep this window open while using the app. Press Ctrl+C to stop.
echo.
set "OPEN=--open-browser"
if defined VEILX_NO_BROWSER set "OPEN="
"%PY%" scripts\run_app.py %OPEN% %*
goto end

:no_pack
echo.
echo  [!] The VEIL-X runtime pack was not found.
echo      Put VEIL-X_runtime_pack.zip next to run_veilx.bat and run it again.
echo      It holds the frozen model and decision files that are not published
echo      on GitHub. Ask the project owner for it.
goto end

:no_uv
echo.
echo  [!] uv is required. Install it from https://docs.astral.sh/uv/ and run
echo      run_veilx.bat again.
goto end

:fail
echo.
echo  [!] Setup failed - see the messages above. Check your internet
echo      connection and free disk space, then run run_veilx.bat again.
echo      If the problem persists, delete the .venv-app folder and retry.

:end
echo.
pause
endlocal
