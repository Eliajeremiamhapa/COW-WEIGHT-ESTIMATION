#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# If you use the standard opencv-python, you'd add apt-get commands here, 
# but opencv-python-headless usually fixes this automatically.