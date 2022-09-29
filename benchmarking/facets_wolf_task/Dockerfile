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

WORKDIR /app
