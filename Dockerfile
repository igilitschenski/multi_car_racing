FROM ubuntu:latest
RUN apt-get -y update

# install python 3.7
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip
RUN python3.9 -m pip install pip
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-setuptools
RUN python3.9 -m pip install pip --upgrade pip

# copy the project directory
ARG main_dir=/home/CSCI527
RUN mkdir -p ${main_dir}/multi_car_racing
WORKDIR ${main_dir}/multi_car_racing

COPY . .

# create venv main folder
ARG env_main_dir=${main_dir}/envs
ARG venv_name=gym_venv
RUN mkdir -p ${env_main_dir}/

# should be run to create venv
RUN apt install python3.9-venv
RUN python3.9 -m venv ${env_main_dir}/${venv_name}

RUN apt-get update && apt-get install -y \
    swig \
    python3.9-dev 
    
# install the requirements
RUN ${env_main_dir}/${venv_name}/bin/pip install --upgrade pip
RUN ${env_main_dir}/${venv_name}/bin/pip install --upgrade setuptools
RUN ${env_main_dir}/${venv_name}/bin/pip install -r requirements.txt

# for opening a video frame 
RUN apt-get update && apt-get install -y \
    python-opengl
RUN apt-get update && apt-get install -y \
    fontconfig
RUN $sudo apt-get install -y \
    xvfb

# cd main_scripts
# xvfb-run -s "-screen 0 1400x900x24" python3.9 train_test_DDPG.py



