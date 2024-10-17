FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://storage.yandexcloud.net/biohack/run.sh -P /workspace

RUN chmod +x /workspace/run.sh
