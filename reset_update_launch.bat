@echo off
title Reset notebooks - Update Python env - Launch JupyterLab
setlocal enabledelayedexpansion

:: Close JupyterLab and Python processes first
echo Closing JupyterLab and Python processes...
taskkill /f /im JupyterLab.exe 2>nul
taskkill /f /im python.exe 2>nul
echo.

:: Navigate to script directory
cd /d "%~dp0"

:: Show current status
echo ========================================
echo Current Git Status:
echo ========================================
git status --short
echo.

:: Check if there are any changes
set has_changes=0
git diff-index --quiet HEAD
if %errorlevel% neq 0 (
    echo WARNING: You have uncommitted changes!
    set has_changes=1
) else (
    git ls-files --others --exclude-standard | findstr . >nul
    if !errorlevel! equ 0 (
        echo WARNING: You have untracked files!
        set has_changes=1
    )
)

:: Only confirm if there are changes
if !has_changes! equ 1 (
    echo.
    echo ========================================
    echo This will RESET the repository to origin/main
    echo ALL local changes will be LOST!
    echo ========================================
    set confirm=y
    set /p confirm="Continue with reset? (y/n) [!confirm!]: "
    if /i "!confirm!"=="n" (
        echo Operation cancelled.
        pause
        exit /b 1
    )
) else (
    echo No local changes detected. Proceeding with update...
    echo.
)

:: Fetch latest from remote
echo.
echo Fetching latest from origin...
git fetch origin
if %errorlevel% neq 0 (
    echo ERROR: Failed to fetch from origin
    pause
    exit /b 1
)

:: Show files that will be deleted
echo.
echo Files/folders that will be deleted:
git clean -n -f -d

:: Reset to origin/main
echo.
echo Resetting to origin/main...
git reset --hard origin/main
if %errorlevel% neq 0 (
    echo ERROR: Failed to reset
    pause
    exit /b 1
)

:: Clean untracked files
echo.
echo Cleaning untracked files...
git clean -f -d

:: Update Python environment
echo.
echo Updating Python environment...
uv sync -p 3.11
if %errorlevel% neq 0 (
    echo ERROR: Failed to update Python environment
    pause
    exit /b 1
)

:: Launch JupyterLab
echo.
echo Launching JupyterLab...
if exist "C:\JupyterLab\JupyterLab.exe" (
    start "" "C:\JupyterLab\JupyterLab.exe"
    echo Done!
) else (
    echo WARNING: JupyterLab.exe not found at C:\JupyterLab\JupyterLab.exe
)

exit /b 0 
