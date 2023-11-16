#FROM tensorflow/tensorflow:1.15.0-gpu-py3
FROM tensorflow/tensorflow:1.15.2-py3
WORKDIR /usr/src/installation
COPY requirements.txt .
RUN pip install --upgrade pip & pip install --no-cache-dir -r requirements.txt
ADD keras-contrib-master .
RUN cd keras-contrib-master & python setup.py install
ADD test /test
ADD app /usr/src/app
WORKDIR /usr/src/app
RUN rm -rf /usr/src/installation