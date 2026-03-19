#!/usr/bin/env bash

apt-get update
apt-get install -y ffmpeg libsndfile1

pip install -r requirements.txt