#!/bin/bash

# sudo dnf update
sudo dnf install libsndfile
python -m pip install --upgrade pip
python3 -m pip install urllib3 --upgrade
python -m pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8080