#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
exec python /opt/video-restore/app/server.py
