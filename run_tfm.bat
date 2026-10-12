@echo off
REM ================================================================
REM  TFM Full TEM Processing Suite - One-Click Launcher
REM  Entry A: GUI (default, double-click)
REM  Entry B: CLI headless (add --cli anywhere)
REM  100% OFFLINE. All data stays local. Never uploaded.
REM ================================================================
setlocal enabledelayedexpansion

set "PYEXE=D:\ProgramData\Anaconda3\python.exe"
set "WORKDIR=%~dp0"
if "%WORKDIR:~-1%"=="\" set "WORKDIR=%WORKDIR:~0,-1%"
set "GUI_SCRIPT=%WORKDIR%\tfm_gui.py"
set "CLI_SCRIPT=%WORKDIR%\tfm_cli_run.py"
set "MODE=GUI"

REM -------- Locate Python --------
where py.exe >nul 2>nul
if %ERRORLEVEL%==0 set "PYEXE=py.exe"
if exist "%PYEXE%" goto PY_OK
for /f "delims=" %%p in ('where python 2^>nul') do set "PYEXE=%%p"
if exist "%PYEXE%" goto PY_OK
echo [ERROR] Python not found. Install Anaconda to D:\ProgramData\Anaconda3\python.exe
pause
exit /b 1
:PY_OK

REM -------- Locate scripts --------
if exist "%GUI_SCRIPT%" goto SCRIPTS_OK
echo [ERROR] Missing: %GUI_SCRIPT%
pause
exit /b 2
:SCRIPTS_OK

REM -------- Arg dispatch (robust, no if-block ambiguity) --------
set "ARGS=%*"
if not defined ARGS goto DISPATCH_DONE
set "TMP=!ARGS:--cli=!"
if not "!TMP!"=="!ARGS!" set "MODE=CLI"
set "TMP=!ARGS:-cli=!"
if not "!TMP!"=="!ARGS!" set "MODE=CLI"
set "TMP=!ARGS:-h=!"
if not "!TMP!"=="!ARGS!" goto HELP
set "TMP=!ARGS:-help=!"
if not "!TMP!"=="!ARGS!" goto HELP
set "TMP=!ARGS:--help=!"
if not "!TMP!"=="!ARGS!" goto HELP
:DISPATCH_DONE

REM -------- DISPATCH via labels (avoids nested if(parentheses) parsing bugs) --------
if /I "%MODE%"=="GUI" goto RUN_GUI
if /I "%MODE%"=="CLI" goto RUN_CLI
echo [ERROR] Unknown MODE=%MODE%. Fallback to GUI.
goto RUN_GUI

:RUN_GUI
echo ================================================================
echo   Launching TFM GUI ...  (close the GUI window to exit)
echo   Python: %PYEXE%
echo   Script: %GUI_SCRIPT%
echo ================================================================
cd /d "%WORKDIR%"
start "" "%PYEXE%" "%GUI_SCRIPT%"
goto END

:RUN_CLI
echo ================================================================
echo   TFM Headless CLI Pipeline  (Preprocess + 1D Inv + Occam Bridge)
echo ================================================================
if not exist "%CLI_SCRIPT%" (
    echo [ERROR] CLI script missing: %CLI_SCRIPT%
    pause
    exit /b 3
)
REM BAT consumed --cli / -cli, remove flags before forwarding to Python
set "CLI_ARGS=!ARGS!"
if defined CLI_ARGS (
    set "CLI_ARGS=!CLI_ARGS: --cli = !"
    set "CLI_ARGS=!CLI_ARGS: -cli = !"
    set "CLI_ARGS=!CLI_ARGS:--cli =!"
    set "CLI_ARGS=!CLI_ARGS:-cli =!"
    REM trailing --cli / -cli at end
    if "!CLI_ARGS:~-5!"=="--cli" set "CLI_ARGS=!CLI_ARGS:~0,-5!"
    if "!CLI_ARGS:~-4!"=="-cli" set "CLI_ARGS=!CLI_ARGS:~0,-4!"
)
cd /d "%WORKDIR%"
"%PYEXE%" "%CLI_SCRIPT%" !CLI_ARGS!
goto END

:HELP
echo Usage:
echo   run_tfm.bat              Launch GUI (default, double-click)
echo   run_tfm.bat  --cli       Run headless full CLI pipeline
echo   run_tfm.bat  -h          Show this help
echo.
echo   Example:
echo     run_tfm.bat --cli --project D:\MyProject\myfile --out D:\output
echo     run_tfm.bat --cli --rhoa-scale 2e6 --lambda-smooth 1.0 --lambda-init 0.1
echo.
echo   100%% OFFLINE.  All data stays on local disk.  Nothing uploaded.
pause
exit /b 0

:END
endlocal
