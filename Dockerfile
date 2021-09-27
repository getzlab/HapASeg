FROM gcr.io/broad-getzlab-workflows/base_image:v0.0.5

WORKDIR /build

# install dependencies
RUN pip install sortedcontainers
RUN git clone https://github.com/getzlab/CApy.git && pip install ./CApy
RUN pip install dask distributed

# install hapaseg
COPY setup.py .
COPY hapaseg ./hapaseg
RUN pip install .

WORKDIR /app
ENV PATH=$PATH:/app
