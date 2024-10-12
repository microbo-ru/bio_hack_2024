# lic
Real-ESRGAN Copyright (c) 2021, Sberbank AI

# weights
```
cd weights
wget https://storage.yandexcloud.net/biohack/hands_final_epoch_checkpoint.pkl
wget https://storage.yandexcloud.net/biohack/RealESRGAN_x8.pth
```

datasphere project job execute -p bt1ct1p5bm3ejocn5k8s -c config.yaml

# python3.11.0, tensorflow2.15.0, keras2.15.0, cuda12.3, keras-cv0.6.0, tensorflow_datasets4.9.6, keras_core0.1.7.

pip install --extra-index-url https://pypi.nvidia.com tensorflow[and-cuda]==2.15.0