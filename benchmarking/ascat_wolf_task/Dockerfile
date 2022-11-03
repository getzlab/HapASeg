FROM rocker/tidyverse:4.2.1

WORKDIR /build
RUN R -e "install.packages('BiocManager'); BiocManager::install(version = '3.15')"
RUN R -e "BiocManager::install(c('GenomicRanges','IRanges'), type = 'source', checkBuilt = TRUE)"
RUN R -e "devtools::install_github('VanLoo-lab/ascat/ASCAT')"

WORKDIR /app
