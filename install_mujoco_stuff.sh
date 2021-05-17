#!/bin/bash

# possibly there is a need to install some python packages via pip,
# but i'm not sure which ones

# you can put your mujoco file in the git repo and push or open firefox and 
# download from your email

# check passing of the license key
if [[ -n ${1} && -s ${1} ]]
then mujoco_key=$1;
else echo "pls provide the path to your mujoco_key.txt license file"
		exit;
fi


# create the necessary directories
cd $HOME
mkdir .local
mkdir .local/bin
mkdir .local/include
mkdir .local/lib
mkdir .local/share

# download mujoco
wget "https://www.roboti.us/download/mujoco200_linux.zip"
mkdir .mujoco
cp $mujoco_key .mujoco
unzip mujoco200_linux .mujoco
cd .mujoco
mv mujoco200_linux mujoco200

# install patchelf 
cd $HOME
git clone "https://github.com/NixOS/patchelf"
cd patchelf
./bootstrap.sh
./configure --prefix $HOME/.local
make
make check
make install


# download libosmesa6-dev
# download libosmesa 
# copy the files into the right directories
# check the files that were downloaded in the vm
# they should be debian files which were then extracted with
# dpkg -x package.deb folder
# afaik .h goes in something and libOSMesa.so should be renamed into that (not the links,
# the thing itself and then sent into .mujoco/mujoco/bin


# set environmental variables in .bashrc
#...
source ~/.bashrc



# install mujoco-py
# maybe you also need to go from source i don't remember
pip3 install mujoco-py
