```
aws s3 cp main.py s3://biohack/
datasphere project job execute -p bt1ct1p5bm3ejocn5k8s -c config.yaml
cat /tmp/datasphere/job_2024-10-17T12:19:18.065072/system.log
```


docker tag dc74aab0ee08 xhoopoe/pytorch:biohack
docker push xhoopoe/pytorch:biohack

```
docker build .
docker tag 13f743e55906 cr.yandex/crpp9388oa059nnfhi8m/microbo:biohack
docker push cr.yandex/crpp9388oa059nnfhi8m/microbo:biohack
```

```
docker run --gpus all -it --rm cr.yandex/crpp9388oa059nnfhi8m/microbo
```

```
aws s3 cp main.py s3://biohack/
```

zip -r hands_v4.zip hands_v4

aws s3 cp hands_v4.zip s3://biohack/

https://storage.yandexcloud.net/biohack/hands_v4.zip


hands_train_job git:(master) ✗ datasphere project job execute -p bt1ct1p5bm3ejocn5k8s -c config.yaml
2024-10-16 19:57:39,571 - [INFO] - logs file path: /tmp/datasphere/job_2024-10-16T19:57:39.565252
2024-10-16 19:57:39,601 - [INFO] - creating job ...
2024-10-16 19:57:41,941 - [INFO] - no files to upload
2024-10-16 19:57:41,941 - [INFO] - created job `bt10b51ibolsgfl7l1km`
2024-10-16 19:57:42,459 - [INFO] - executing job ...
2024-10-16 20:00:56,999 - [INFO] - job link: https://datasphere.yandex.cloud/communities/bt1r6tal3de7mn1l3gsg/projects/bt1ct1p5bm3ejocn5k8s/job/bt10b51ibolsgfl7l1km
2024-10-16 20:00:56,999 - [ERROR] - Operation error occurred
        logs file path: /tmp/datasphere/job_2024-10-16T19:57:39.565252
        headers
                x-request-id: bd33aac1-8097-4616-ad07-7493525f4f59
                date: Wed, 16 Oct 2024 16:57:42 GMT
                server: ycapi
                x-server-trace-id: dd8318a4eaa859aa:cf34fdc4a4738b76:dd8318a4eaa859aa:1
2024-10-16 20:00:57,003 - [INFO] - see job log files at /tmp/datasphere/job_2024-10-16T19:57:39.565252
2024-10-16 20:00:57,003 - [INFO] -   * stdout.txt  (0.0B)
2024-10-16 20:00:57,004 - [INFO] -   * system.log  (1.0KB)
2024-10-16 20:00:57,004 - [INFO] -   * gpu_stats.tsv  (0.0B)
2024-10-16 20:00:57,004 - [INFO] -   * docker_stats.tsv  (0.0B)
2024-10-16 20:00:57,005 - [INFO] -   * stderr.txt  (0.0B)
2024-10-16 20:00:57,005 - [INFO] -   * log.txt  (10.1KB)
Traceback (most recent call last):
  File "/home/vladimir/anaconda3/envs/gan/bin/datasphere", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/vladimir/anaconda3/envs/gan/lib/python3.11/site-packages/datasphere/main.py", line 210, in main
    args.func(args, sdk)
  File "/home/vladimir/anaconda3/envs/gan/lib/python3.11/site-packages/datasphere/main.py", line 17, in execute
    sdk._execute(args)
  File "/home/vladimir/anaconda3/envs/gan/lib/python3.11/site-packages/datasphere/sdk.py", line 120, in _execute
    self.client.wait_for_completion(op, execute_call, shutdown_params)
  File "/home/vladimir/anaconda3/envs/gan/lib/python3.11/site-packages/datasphere/client.py", line 195, in wait_for_completion
    op = self._poll_operation(op, call_which_created_op, graceful_shutdown_params)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vladimir/anaconda3/envs/gan/lib/python3.11/site-packages/datasphere/client.py", line 224, in _poll_operation
    raise OperationError(op, call_which_created_op)
datasphere.client.OperationError: Operation returned error:
        status=INTERNAL
        details=Cannot create env: Image pull failed with error Status 404: {"message":"pull access denied for cr.yandex/crpntcvhsvn7ju2anpul/microbo, repository does not exist or may require 'docker login': denied: Permission denied ; requestId = 0a263208-1801-4c1c-8ea1-15643197007c"}