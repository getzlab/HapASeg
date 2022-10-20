FROM gcr.io/broad-getzlab-workflows/base_image:v0.0.5

WORKDIR /build

# install dependencies (SAMtools and BCFtools and htslib already in standard docker)
# mosdepth
RUN echo '#!/bin/bash' > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo
RUN wget https://github.com/brentp/mosdepth/releases/download/v0.3.3/mosdepth && \
    cp mosdepth /usr/local/bin && chmod +x /usr/local/bin/mosdepth

# shapeit
RUN wget https://mathgen.stats.ox.ac.uk/genetics_software/shapeit/shapeit.v2.r904.glibcv2.17.linux.tar.gz && \
    tar -zxvf shapeit.v2.r904.glibcv2.17.linux.tar.gz && rm shapeit.v2.r904.glibcv2.17.linux.tar.gz && \
    cp ./shapeit.v2.904.3.10.0-693.11.6.el7.x86_64/bin/shapeit /usr/local/bin

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;
    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    rm /etc/ssl/certs/java/cacerts && \
    update-ca-certificates --fresh;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# picard
RUN wget https://github.com/broadinstitute/picard/releases/download/2.27.4/picardcloud.jar && \
    cp picardcloud.jar /usr/local/bin/ && ln -s /usr/local/bin/picardcloud.jar /usr/local/bin/picard.jar
ENV HATCHET_PATHS_PICARD /usr/local/bin/
RUN export HATCHET_PATHS_PICARD

# Install cbc Solver
# HATCHet recommends using Gurobi, but using its academic license in a cluster is complicated. cbc will work (a little slower though)
RUN apt-get install -y coinor-cbc coinor-libcbc-dev

# HATCHet
RUN pip install -U pip && pip install -U setuptools
ENV HATCHET_BUILD_NOEXT 1
RUN export HATCHET_BUILD_NOEXT
ENV HATCHET_COMPUTE_CN_SOLVER cbc
RUN export HATCHET_COMPUTE_CN_SOLVER
RUN git clone https://github.com/raphael-group/hatchet.git && cd hatchet && pip install .

#
# CLEAN UP {{{

RUN rm -rf /build/*

# }}}

#
# CONFIGURE ENVIRONMENT {{{

ENV LD_LIBRARY_PATH=/usr/local/lib

# }}}