# Dockerfile for setting up a custom development environment
# based on Ubuntu20.04 with NVIDIA CUDA.
# Command to buid image: 
# docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) .

# Use the base image with NVIDIA CUDA and cuDNN pre-installed
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set build arguments for user and group IDs
ARG UID
ARG GID
ARG USERNAME=user
ENV HOME /home/${USERNAME}

# Set the timezone to Europe/Warsaw
RUN ln -fs /usr/share/zoneinfo/Europe/Warsaw /etc/localtime

# Install required packages
RUN apt-get update && \
    apt-get install -y nano git wget unzip openssh-server python3.8 \ 
    python3.8-dev python3-pip && \
    ln -sf /usr/bin/python3.8 /usr/bin/python

# Install Python packages using pip
RUN pip install spikingjelly==0.0.0.0.14 scikit-learn wandb \
    cupy-cuda11x numpy==1.23.5

# Set root password
RUN echo 'root:root' | chpasswd

# Create a non-root user and set up its permissions
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -u ${UID} -g ${GID} -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME}:${USERNAME}" | chpasswd && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

# Switch to the non-root user and set up SSH key and working directory
# USER ${USERNAME}
# RUN mkdir -p /home/plgkrzysjed1/.ssh
# RUN mkdir -p ${HOME}/workbench
# COPY /.ssh/id_rsa.pub /home/plgkrzysjed1/.ssh/
# COPY /.ssh/authorized_keys /home/plgkrzysjed1/.ssh/

# Start SSH service
USER root
RUN service ssh start

# Expose SSH port for remote connections
EXPOSE 22

# Start SSH server to keep the container running
CMD ["/usr/sbin/sshd", "-D"]
