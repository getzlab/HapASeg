FROM rocker/tidyverse:3.6.1

WORKDIR /build
ENV HTSLIB_VERSION 1.5
ENV HTSTOOLS_VERSION 0.1.1
ENV FACETSSUITE_VERSION Rpackagev2

RUN wget https://github.com/mskcc/facets/archive/v0.5.14.zip -O facets_v0.5.14.zip && \
    unzip facets_v0.5.14.zip && \
    rm -f facets_v0.5.14.zip && \
    cd facets-0.5.14 && \
    Rscript -e "devtools::install()"

# download utils to install htslib
RUN sudo apt-get update && sudo apt-get -y install gcc make libbz2-dev zlib1g-dev libncurses5-dev \
libncursesw5-dev liblzma-dev bzip2
RUN wget https://github.com/samtools/htslib/releases/download/1.16/htslib-1.16.tar.bz2 && \
tar -vxjf htslib-1.16.tar.bz2 && cd htslib-1.16 && ./configure --prefix /build && \
make && make install

# now install snp-pileup 
RUN cd /build/facets-0.5.14/inst/extcode && sudo ldconfig && g++ -std=c++11 -I /build/include snp-pileup.cpp \
-L /build/lib -lhts -Wl,-rpath=/build/lib -o snp-pileup
ENV PATH "$PATH:/build/facets-0.5.14/inst/extcode"
WORKDIR /app
