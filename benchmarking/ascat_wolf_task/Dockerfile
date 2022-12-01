FROM rocker/tidyverse:4.2.1

WORKDIR /build
RUN R -e "install.packages('BiocManager'); BiocManager::install(version = '3.15')"
RUN R -e "BiocManager::install(c('GenomicRanges','IRanges'), type = 'source', checkBuilt = TRUE)"
RUN R -e "devtools::install_github('VanLoo-lab/ascat/ASCAT')"

# get utils necessary for allelecounter install
RUN sudo apt-get update && sudo apt-get -y install gcc make libbz2-dev zlib1g-dev libncurses5-dev libncursesw5-dev liblzma-dev bzip2 curl
RUN git clone https://github.com/cancerit/alleleCount.git && cd alleleCount && ./setup.sh /build/alleleCounter

ENV PATH "$PATH:/build/alleleCounter/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/build/alleleCounter/lib"

WORKDIR /app
