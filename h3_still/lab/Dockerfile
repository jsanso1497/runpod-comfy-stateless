# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless:h3-still-c8ad356ff6f7d2ca327c0395b21ec608062b900c
FROM ${BASE_IMAGE}
ARG LAB_PROFILE=stable
WORKDIR /opt/h3-still/lab
COPY . .
RUN python -m unittest discover -s tests -v \
    && python assemble.py --profile "$LAB_PROFILE" \
    && python -m compileall -q /opt/ComfyUI/custom_nodes/H3_Studio_Lab \
    && python -m pip freeze > /opt/h3-still/lab/build-pip-freeze.txt
WORKDIR /workspace
EXPOSE 8188 8888
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "/opt/h3-still/lab/launch.py"]
