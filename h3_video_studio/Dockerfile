# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless:h3-still-c8ad356ff6f7d2ca327c0395b21ec608062b900c
FROM ${BASE_IMAGE}
WORKDIR /opt/h3-still
COPY h3_still/lab/ /opt/h3-still/lab/
COPY h3_video_studio/ /opt/h3-still/video_studio/
RUN python -m unittest discover -s /opt/h3-still/video_studio/tests -v \
    && python /opt/h3-still/video_studio/assemble.py \
    && python -m pip freeze > /opt/h3-still/video_studio/build-pip-freeze.txt
ENV H3V_PRELOAD=hq
WORKDIR /workspace
EXPOSE 8188 8888
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "/opt/h3-still/video_studio/launch.py"]
