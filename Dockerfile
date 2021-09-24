FROM gcr.io/broad-getzlab-workflows/base_image:v0.0.5

WORKDIR /build
# install hapaseg
COPY setup.py .
COPY hapaseg ./hapaseg
RUN pip install .

WORKDIR /app
ENV PATH=$PATH:/app
