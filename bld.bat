rem create required folders
if not exist "%SCRIPTS%" mkdir "%SCRIPTS%"

rem copy post-install and pre-uninstall scripts
copy "%SRC_DIR%\qats-post-install.py" "%SCRIPTS%"
copy "%SRC_DIR%\qats-pre-uninstall.py" "%SCRIPTS%"

rem added to prevent setuptools from creating an egg-info directory because they do not interact well with conda
rem "%PYTHON%" setup.py install
"%PYTHON%" setup.py install --single-version-externally-managed --record record.txt

if errorlevel 1 exit 1
