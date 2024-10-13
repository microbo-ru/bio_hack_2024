```
datasphere project job execute -p bt1ct1p5bm3ejocn5k8s -c config.yaml
```

```
docker build .
docker tag 13f743e55906 cr.yandex/crpntcvhsvn7ju2anpul/microbo:biohack
docker push cr.yandex/crpntcvhsvn7ju2anpul/microbo:biohack
```

```
docker run --gpus all -it --rm cr.yandex/crpntcvhsvn7ju2anpul/microbo
```