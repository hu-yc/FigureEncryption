# README

## Description
本项目致力于通过GAN，完成对图像的加密，并输出加密后的图像和性能评估结果。

## Deployment
本项目可通过docker镜像部署
镜像见文件夹`img`

```docker
# under the project directory
# loading images
docker load -i img/encrypt_model.tar
docker load -i img/encrypt_nginx.tar
# intializing containers
docker-compose up -d
```

## API Doc

本项目提供交互的api，代码样例如下：
```python
import requests

def upload_img_to_request(img_path):
    img = open(img_path, "rb")
    response = requests.post(
        "http://localhost:8099/model/encrypt", 
        files={'img': img}
    )
    json_response = response.json()
    return json_response
```
返回结果样例
```json
{
    "encrypt_fig_path": "http://127.0.0.1:8099/outputs/encrypted_2f18a6f80faa00a1c78d13249ddf32b2_20231116050623.png",
    "performance_plot_path": "http://127.0.0.1:8099/outputs/performance_2f18a6f80faa00a1c78d13249ddf32b2_20231116050624.png"
}
```
另，可以通过postman调用api, 配置路径见 
```text
-- postman_config
    -- API.postman_collection.json
    -- encrypt_variables.postman_environment.json
```
在postman中导入配置文件后，请启用环境变量`encrypt_variables`，并将`img_path`设置为需要加密的图片路径，然后发送请求即可。
