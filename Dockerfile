FROM gcr.io/broad-getzlab-workflows/base_image:v0.0.5

WORKDIR /build

# install dependencies
RUN pip install sortedcontainers
ARG cache_invalidate=10
RUN git clone https://github.com/getzlab/CApy.git && pip install ./CApy
RUN pip install distinctipy
RUN pip install kneed
RUN git clone -b oliver_sim_changes https://github.com/getzlab/cnv_suite.git && pip install ./cnv_suite

# install hapaseg
ADD benchmarking/ ./benchmarking
COPY setup.py .
COPY hapaseg ./hapaseg
COPY hapaseg_local ./hapaseg_local
RUN pip install .

WORKDIR /app
ENV PATH=$PATH:/app
