@echo off
@echo Removing QATS start menu and desktop shortcuts.>> %PREFIX%\.messages.txt
rem call python "%PREFIX%\Scripts\qats-pre-uninstall.py"
call python "%~dp0\qats-pre-uninstall.py"
