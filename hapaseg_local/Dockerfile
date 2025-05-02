FROM ubuntu:20.04 AS base

WORKDIR build

# update apt
ARG version
RUN apt-get update

#
# UTILITIES 

RUN apt-get install -y build-essential vim nano git wget curl ssed htop

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get -y install tzdata

#
# PYTHON {{{

# Python 3.8
RUN apt-get install -y python3.8 python3.8-dev && ln -s /usr/bin/python3.8 /usr/bin/python

# pip
RUN apt-get install -y python3.8-distutils && curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | python
               
# packages
RUN pip install pandas numpy scipy matplotlib crcmod
RUN pip install firecloud-dalmatian

# }}}

#
# GCLOUD {{{

RUN mkdir /gcsdk && \
  wget -O gcs.tgz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-318.0.0-linux-x86_64.tar.gz && \
  tar xzf gcs.tgz -C /gcsdk && \
  /gcsdk/google-cloud-sdk/install.sh --usage-reporting false --path-update true --quiet && \
  ln -s /gcsdk/google-cloud-sdk/bin/* /usr/bin

# }}}

#
# SAMTOOLS {{{

RUN apt-get install -y libssl-dev libcurl4-openssl-dev liblzma-dev libbz2-dev libncurses5-dev zlib1g-dev
RUN wget https://github.com/samtools/htslib/releases/download/1.16/htslib-1.16.tar.bz2 && \
    tar -xf htslib-1.16.tar.bz2 && rm htslib-1.16.tar.bz2 && cd htslib-1.16 && \
    ./configure --enable-libcurl --enable-s3 --enable-plugins --enable-gcs && \
    make && make install && make clean
RUN wget https://github.com/samtools/samtools/releases/download/1.16/samtools-1.16.tar.bz2 && \
    tar -xf samtools-1.16.tar.bz2 && rm samtools-1.16.tar.bz2 && cd samtools-1.16 && \
    ./configure --with-htslib=system && make && make install && make clean
RUN wget https://github.com/samtools/bcftools/releases/download/1.16/bcftools-1.16.tar.bz2 && \
    tar -xf bcftools-1.16.tar.bz2 && rm bcftools-1.16.tar.bz2 && cd bcftools-1.16 && \
    ./configure --with-htslib=system && make && make install && make clean


# seqlib

RUN git clone https://github.com/julianhess/SeqLib.git && \
  cd SeqLib && git checkout origin/jhess && git submodule update --init && \
  ./configure LDFLAGS="-lcurl -lcrypto" && \
  make CXXFLAGS='-std=c++11' && make install && \
  mv lib/* /usr/local/lib
RUN cd SeqLib && mkdir -p /usr/local/include/SeqLib && mv SeqLib bwa fermi-lite json /usr/local/include
ENV INCLUDE=/usr/local/lib/SeqLib

# }}}

#
# CLEAN UP {{{

RUN rm -rf /build/*

# }}}

#
# CONFIGURE ENVIRONMENT {{{

ENV LD_LIBRARY_PATH=/usr/local/lib

# }}}

#
# COVCOLLECT {{{
RUN git clone --recurse-submodules https://github.com/getzlab/covcollect.git && cd covcollect/src && make
WORKDIR /app
ENV PATH=$PATH:/app
RUN cp /build/covcollect/src/covcollect /app && rm -rf /build/*

#}}}

#
# MUTECT1 {{{
  WORKDIR /mutect_build

  #
  # prereqs
  
  # install JDK
  RUN wget https://files.liferay.com/mirrors/download.oracle.com/otn-pub/java/jdk/7u80-b15/jdk-7u80-linux-x64.tar.gz && \
   tar xzf jdk-7u80-linux-x64.tar.gz && mkdir /usr/local/java && mv jdk1.7.0_80/ /usr/local/java
  
  # install Maven
  RUN apt-get update && apt install -y maven
  
  #
  # get/install MuTect
  ENV JAVA_HOME=/usr/local/java/jdk1.7.0_80
  
  # prereq: install GATK
  RUN git clone https://github.com/broadgsa/gatk-protected.git && cd gatk-protected && \
    git reset --hard 3.1 
  RUN cd gatk-protected && sed -i '401,557d' pom.xml && mvn -Ddisable.queue -Dhttps.protocols=TLSv1.2 install
  
  # install MuTect
  ARG cache_invalidate=xxx
  RUN git clone https://github.com/getzlab/mutect.git && cd mutect && \
     mvn -Ddisable.queue -Dhttps.protocols=TLSv1.2 verify
  
  # symlink to /app
  WORKDIR /app
  RUN ln -s /mutect_build/mutect/target/mutect-1.1.7.jar mutect.jar

#}}}

#
# EAGLE2 {{{
  WORKDIR /build
  SHELL ["/bin/bash", "-c"]
  RUN wget https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/Eagle_v2.4.1.tar.gz && tar xzf Eagle_v2.4.1.tar.gz && mv Eagle_v2.4.1/{eagle,tables} /app && rm -rf *
  
#}}}

RUN apt-get install -y parallel


ARG cache_invalidate=13
#
# HapASeg {{{
RUN pip install sortedcontainers click

RUN pip install distinctipy
RUN pip install kneed
RUN git clone -b oliver_sim_changes https://github.com/getzlab/cnv_suite.git && pip install ./cnv_suite

# install hapaseg
ADD benchmarking/ ./benchmarking
COPY setup.py .
COPY hapaseg ./hapaseg
COPY hapaseg_local ./hapaseg_local
RUN pip install .

#}}}

#
# Het pulldown {{{
  #RUN git clone https://github.com/getzlab/CApy.git && cd CApy && git checkout eb95ce4 && pip install .
  RUN git clone https://github.com/oliverpriebe/CApy.git && pip install ./CApy
  RUN git clone https://github.com/getzlab/het_pulldown_from_callstats_TOOL.git && cd het_pulldown_from_callstats_TOOL && cp hetmodels.py hetpull.py /app
  
  #}}}
