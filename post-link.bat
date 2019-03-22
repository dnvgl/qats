@echo off
@echo Installing QATS start menu and desktop shortcuts>> %PREFIX%\.messages.txt
rem call python "%PREFIX%\Scripts\qats-post-install.py"
call python "%~dp0\qats-post-install.py"
if errorlevel 1 (
    @echo Failed to run Python to install start menu and desktop shortcuts.>> %PREFIX%\.messages.txt
    exit 1
)
