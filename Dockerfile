FROM elasticsearch:7.4.2

ENV discovery.type=single-node

USER root

COPY . .

RUN yum install -y https://repo.ius.io/ius-release-el7.rpm && \
    yum -y update && \
    yum -y install gcc openssl-devel bzip2-devel libffi-devel wget make xz-devel && \
    wget https://www.python.org/ftp/python/3.8.7/Python-3.8.7.tgz && \
    tar xzf Python-3.8.7.tgz && \
    cd Python-3.8.7 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd ..  && \
    python3.8 -m pip install cython && \
    python3.8 -m pip install numpy && \
    python3.8 -m pip install -r requirements.txt

RUN echo "path.data: /index/elasticsearch/" >> /usr/share/elasticsearch/config/elasticsearch.yml

RUN sed -i -e '2inohup python3.8 app.py &\' /usr/local/bin/docker-entrypoint.sh