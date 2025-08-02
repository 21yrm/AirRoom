FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    tmux \
    python3.8 python3.8-dev python3.8-venv python3-distutils \
    python3-opencv ca-certificates git wget sudo ninja-build \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3 get-pip.py --user && \
    rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install --user torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --user opencv-python
RUN pip install --user pyyaml
RUN pip install --user json_tricks
RUN pip install --user yacs
RUN pip install --user scikit-learn
RUN pip install --user pandas
RUN pip install --user timm==0.4.12
RUN pip install --user numpy==1.23.5
RUN pip install --user einops
RUN pip install --user fvcore
RUN pip install --user transformers
RUN pip install --user sentencepiece
RUN pip install --user ftfy
RUN pip install --user regex
RUN pip install --user nltk
RUN pip install --user vision-datasets==0.2.2
RUN pip install --user pycocotools
RUN pip install --user diffdist
RUN pip install --user pyarrow
RUN pip install --user cityscapesscripts
RUN pip install --user shapely
RUN pip install --user scikit-image
RUN pip install --user mup
RUN pip install --user gradio==3.35.2
RUN pip install --user scann
RUN pip install --user kornia==0.6.4
RUN pip install --user torchmetrics==0.6.0
RUN pip install --user progressbar
RUN pip install --user pillow==9.4.0
RUN pip install --user natsort
RUN pip install --user plyfile
RUN pip install --user faiss-gpu
RUN pip install --user onedrivedownloader
RUN pip install --user fast-pytorch-kmeans
RUN pip install --user gdown
RUN pip install --user psutil

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Download Semantic-SAM weights
RUN mkdir -p /home/appuser/models
RUN wget -O /home/appuser/models/swinl_only_sam_many2many.pth https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth

WORKDIR /home/appuser/detectron2_repo

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
# --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
# --input input.jpg --output outputs/ \
# --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

USER root

RUN wget https://github.com/mikefarah/yq/releases/download/v4.34.1/yq_linux_amd64 -O /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

USER appuser
