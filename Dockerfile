# FROM defines the base image
FROM nvidia/cuda:10.1-cudnn7-devel
# nvidia/cuda:10.1-cudnn7-devel
# RUN executes a shell command
# You can chain multiple commands together with && 
# A \ is used to split long lines to help with readability
# This particular instruction installs the source files 
# for deviceQuery by installing the CUDA samples via apt
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-$CUDA_PKG_VERSION
    # rm -rf /var/lib/apt/lists/* && \
RUN apt-get install -y python3.6-dev python3-pip mc git emacs-nox libxrender1 graphviz && pip3 install --upgrade pip setuptools
COPY tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl /root
COPY macprogpu.req /root
WORKDIR /root
RUN python3 -m pip install --upgrade tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl && \
    python3 -m pip install jupyter keras six wheel mock 'future>=0.17.1' && \
    python3 -m pip install keras_applications --no-deps && \
    python3 -m pip install keras_preprocessing --no-deps && \
    python3 -m pip install -r macprogpu.req && \
    ln -s /usr/bin/python3 /usr/bin/python
# set the working directory
# DONE compile tenzorflow weith cuda compute 3.0 
# jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
ENTRYPOINT ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0" , "--allow-root"]