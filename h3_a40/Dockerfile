# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless:h3-still-c8ad356ff6f7d2ca327c0395b21ec608062b900c
FROM ${BASE_IMAGE}
WORKDIR /opt/h3-still/a40
COPY . .
RUN python -m unittest discover -s tests -v \
    && python assemble.py \
    && python -m compileall -q /opt/h3-still/a40 \
    && python -m pip freeze > /opt/h3-still/a40/build-pip-freeze.txt
ENV H3_STILL_DATA_ROOT=/workspace/h3-still
WORKDIR /workspace
EXPOSE 8188 8888
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "/opt/h3-still/a40/runtime.py", "launch"]
