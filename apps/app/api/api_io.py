import os
from flask import Flask, jsonify, request

from utils.utils_log import LogFactory
from model.Functional import CallResNet, ImageInteract

app_server = Flask(__name__)

log = LogFactory.get_log('audit')


def encrypt_resnet(img_path):
    call_resnet = CallResNet()
    image_interact_tool = ImageInteract(img_path)
    random_figure = call_resnet.random_image_generator()
    encrypt_period, encrypt_plot_path = image_interact_tool.encrypt_fig(random_figure)
    decrypt_period, decrypt_plot_path = image_interact_tool.decrypt_fig(random_figure)
    entropy = image_interact_tool.cal_entropy()
    performance_plot_path = image_interact_tool.plot_performance_indicator(encrypt_period, decrypt_period, entropy)
    return encrypt_plot_path, decrypt_plot_path, performance_plot_path


@app_server.route('/model/encrypt', methods=['POST'])
def encrypt():
    # obtain the image from request body
    img = request.files['img']
    # save the image to "/tmp" directory
    img_path = '/tmp/{}'.format(img.filename)
    img.save(img_path)
    log.info("[encrypt] save the image to img_path: {}".format(img_path))
    encrypt_path, decrypt_path, performance_path = encrypt_resnet(img_path)
    host_ip = os.getenv('HOST_IP', '0.0.0.0')
    base_url = 'http://{}:8099'.format(host_ip)
    log.info("[encrypt] encrypt figure path: {}, performance figure path: {}".format(encrypt_path, performance_path))
    return jsonify({
        'encrypt_fig': base_url+'{}'.format(encrypt_path),
        'decrypt_fig': base_url+'{}'.format(decrypt_path),
        'performance_plot': base_url+'{}'.format(performance_path)
    })
