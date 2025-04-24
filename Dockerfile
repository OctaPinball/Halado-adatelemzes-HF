#Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

#Install Python3 and pip if not already installed
RUN apt-get update && apt-get install -y python3-pip

#Upgrade pip
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
RUN pip3 install --upgrade pip
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

#Set up a working directory
WORKDIR /workspace

CMD ["bash"]