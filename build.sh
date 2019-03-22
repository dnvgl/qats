#!/usr/bin/env bash

# copy post-install and pre-uninstall scripts
# note 1: Environment variable SCRIPTS is only available when building on Windows, hence needs to be defined explicitly.
# note 2: During convert to win-64 later, the 'bin' folder will be renamed to 'Scripts', so at this point the 'Scripts'
#         folder must be empty. The post-install and pre-uninstall scripts are therefore copied to 'bin' folder, in
#         order to end up in Scripts folder on Windows.
SCRIPTS=$PREFIX/bin
mkdir -p "$SCRIPTS"
cp "$SRC_DIR/qats-post-install.py" "$SCRIPTS"
cp "$SRC_DIR/qats-pre-uninstall.py" "$SCRIPTS"

# $PYTHON setup.py install
$PYTHON setup.py install --single-version-externally-managed --record record.txt
