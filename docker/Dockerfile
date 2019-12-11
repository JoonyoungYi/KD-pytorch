FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN pip install --upgrade pip
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install git+https://github.com/JoonyoungYi/fancyimpute.git@1608f493b26d8418d69c54661ef0688d7c11f78d#egg=fancyimpute
RUN pip install tqdm==4.35.0
RUN pip install torch==1.2.0
RUN pip install torchvision==0.4.0
