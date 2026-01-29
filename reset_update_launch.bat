echo off
title Reset notebooks - Update Python env - Launch JupyterLab

taskkill /f /im JupyterLab.exe
taskkill /f /im python.exe

cd c:\users\svc_neuropix\documents\github\np_notebooks

echo git reset
git reset --hard -q

echo git clean
@REM first display the files without deleting, in case the user isn't aware of what this script does
@REM git clean -n -f -d

@REM y for yes: default behavior is to delete untracked files in np_notebooks
set clean=y
set /p clean=The above files/folders will be deleted. Clean? (y/n) [%clean%]
IF not %clean%==n (
git clean -f -d
)

echo switch to main
git checkout origin/main
git pull origin main

uv sync -p 3.11

start C:\JupyterLab\JupyterLab.exe

exit /b 
