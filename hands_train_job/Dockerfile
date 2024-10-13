FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

COPY run.sh /workspace