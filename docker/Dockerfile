FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
ARG NJ=1
RUN echo "NJ=${NJ}"
# switch to TW mirror
RUN sed -i "s:archive.:tw.archive.:g" /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y aptitude apt-utils
RUN apt-get install -y build-essential git cmake xterm clang-3.9 clang++-3.9 libomp-dev
ENV TERM xterm-256color
ENV DEBIAN_FRONTEND noninteractive
ENV CC=clang-3.9
ENV CXX=clang++-3.9

# install Google Test
RUN apt-get install -y libgtest-dev
WORKDIR /usr/src/gtest
RUN cmake .
RUN make -j ${NJ}
RUN make -j ${NJ} install

# install ArrayFire with cpu+cuda backend
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 && \
    ln -s /usr/lib/libcuda.so.1 /usr/lib/libcuda.so
WORKDIR /root
RUN apt-get install -y libfftw3-dev libblas-dev liblapack-dev liblapacke-dev
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y pkg-config libfreetype6-dev libboost-all-dev libglfw3-dev libgl1-mesa-dev
RUN apt-get install -y libglm-dev wget
RUN wget --quiet http://arrayfire.com/arrayfire_source/arrayfire-full-3.6.2.tar.bz2
RUN tar xf arrayfire-full-3.6.2.tar.bz2
RUN mv arrayfire-full-3.6.2 arrayfire
RUN mkdir -p arrayfire/build
WORKDIR /root/arrayfire/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_OPENCL=OFF -DBUILD_CUDA=ON -DBUILD_CPU=ON -DBUILD_TEST=OFF
RUN make -j ${NJ} 
RUN make -j ${NJ}  install

# install GLog
RUN apt-get install -y libgoogle-glog-dev

# install flashlight
# 1. cereal
WORKDIR /root
RUN apt-get install -y gcc-multilib g++-multilib libopenmpi-dev
RUN wget --quiet https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz
RUN tar xf v1.2.2.tar.gz && ln -s cereal-1.2.2 cereal
RUN mkdir -p cereal/build
WORKDIR /root/cereal/build
# uncommented next line when using gcc/g++
# RUN cmake .. -DTHREAD_SAFE=ON -DCMAKE_CXX_FLAGS="-Werror=implicit-fallthrough=0 -pthread"
RUN cmake .. -DTHREAD_SAFE=ON -DCMAKE_CXX_FLAGS="-pthread"
RUN make -j ${NJ} 
RUN make -j ${NJ} install
# 2. flashlight
WORKDIR /root
RUN apt-get install -y unzip
RUN wget --quiet https://github.com/facebookresearch/flashlight/archive/master.zip
RUN unzip master.zip && ln -s flashlight-master flashlight
RUN mkdir -p flashlight/build
WORKDIR /root/flashlight/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA -DCMAKE_POSITION_INDEPENDENT_CODE=ON
RUN make -j ${NJ} 
RUN make -j ${NJ} install

#install 
#kenlm
WORKDIR /root
RUN apt-get install -y libboost-all-dev libbz2-dev libeigen3-dev liblzma-dev libz-dev
RUN wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
RUN mkdir -p kenlm/build
WORKDIR /root/kenlm/build
RUN CC=gcc CXX=g++ cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
RUN make -j ${NJ}  &&  make -j ${NJ} install
RUN cp lib/* /usr/local/lib/
WORKDIR /root/kenlm/include/util
RUN ln -s ../../util/double-conversion double-conversion

WORKDIR /root
RUN apt-get install -y libogg-dev libvorbis-dev libflac-dev libsndfile1-dev
RUN git clone --recursive https://github.com/alex-ht/wav2letter.git
RUN cd wav2letter && git checkout cblas
RUN mkdir -p wav2letter/build
WORKDIR /root/wav2letter/build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
          -DCRITERION_BACKEND=CUDA \
          -DCMAKE_INCLUDE_PATH=/root/kenlm/include \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          ..
RUN make -j ${NJ} 
