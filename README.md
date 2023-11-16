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

本项目提供交互的api，**Python代码样例**如下：
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

另，可以**通过postman调用api**, 配置路径见 
```text
-- encrypt_fig
    -- postman_config
        -- API.postman_collection.json
        -- encrypt_variables.postman_environment.json
```
在postman collection中导入`API.postman_collection.json`，

在postman environment中导入`encrypt_variables.postman_environment.json`，请启用环境变量集合`encrypt_variables`

在collection中的API集合中选择`encrypt`请求，将body中的`img`参数设置为需要加密的图片路径，然后发送请求即可。

**返回结果样例**
```json
{
    "encrypt_fig": "http://127.0.0.1:8099/outputs/encrypted_2f18a6f80faa00a1c78d13249ddf32b2_20231116050623.png",
    "performance_plot": "http://127.0.0.1:8099/outputs/performance_2f18a6f80faa00a1c78d13249ddf32b2_20231116050624.png"
}
```
`encrypt_fig`展示的是加密后的图片，`performance_plot`展示的是加密性能评估结果。