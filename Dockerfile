FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev curl wget unzip vim git iputils-ping screen iproute2 tmux && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools wheel && \
    pip3 install  dgl -f https://data.dgl.ai/wheels/repo.html && \
    pip3 install  dglgo -f https://data.dgl.ai/wheels-test/repo.html && \
    pip3 install ipython-ngql pyvis jupyterlab

EXPOSE 8888

WORKDIR /workspace

CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token='nebula'", "--ServerApp.terminado_settings={'shell_command': ['/bin/bash']}", "--ServerApp.allow_remote_access=True", "--ServerApp.allow_origin='*'"]

# docker build -t vesoft/nebula_dgl:v1 .
# docker run -it --gpus all -v $(pwd):/workspace -p 48888:8888 --name nebula-dgl nebula_dgl:v1
