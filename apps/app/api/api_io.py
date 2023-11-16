from flask import Flask, jsonify, request

from utils.utils_log import LogFactory
from model.functional import CallResNet

app_server = Flask(__name__)

log = LogFactory.get_log('audit')


def encrypt_resnet(img_path):
    call_resnet = CallResNet(img_path)
    random_seed = call_resnet.initialized_seed()
    encrypt_period, encrypt_plot_path = call_resnet.encrypt_fig(random_seed)
    entropy, params = call_resnet.cal_entropy()
    decrypt_period, decrypt_plot_path = call_resnet.decrypt_fig(random_seed)
    performance_plot_path = call_resnet.estimate_performance(encrypt_period, decrypt_period, entropy, params)
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
    log.info("[encrypt] encrypt figure path: {}, performance figure path: {}".format(encrypt_path, performance_path))
    return jsonify({
        'encrypt_fig_path': 'http://127.0.0.1:8099{}'.format(encrypt_path),
        'performance_plot_path': 'http://127.0.0.1:8099{}'.format(performance_path)
    })
