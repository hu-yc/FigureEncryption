# README

## Description
本项目致力于通过GAN，完成对图像的加密，并输出加密后的图像和性能评估结果。

## Deployment
本项目可通过docker镜像部署
镜像见文件夹`img`

```docker
# under the img directory
docker load -i XXX.tar
# under the project directory
docker-compose up -d
```

## API Doc
本项目提供交互的api，代码样例如下：
```python
import requests

def upload_img_to_request(img_path):
    img = open(img_path, "rb")
    response = requests.post("http://127.0.0.1:8111/encrypt", files={'img': img})
    json_response = response.json()
    return json_response
```
