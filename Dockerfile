FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    build-essential \
    gfortran \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R
RUN apt-get update && \
    apt-get install -y r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CRAN packages
RUN R -e "install.packages(c( \
    'torch', \
    'luz', \
    'cuda.ml', \
    'remotes' \
    ), repos='https://cloud.r-project.org')"

# Force torch to use CUDA 12.8
ENV CUDA=12.8
ENV TORCH_INSTALL=1

# Install LibTorch / Lantern
RUN R -e "torch::install_torch(force = TRUE)"

# Verify installation
RUN R -e "library(torch); print(torch::cuda_is_available())"

RUN Rscript -e "install.packages(c('data.table', 'Matrix', 'glmnet', 'ggplot2', 'caret', 'xgboost'), repos='https://cloud.r-project.org')"

WORKDIR /app

COPY outputs/ ./outputs/
COPY R/ ./R/
COPY data/ ./data/
COPY docker/entrypoint.sh /entrypoint.sh
COPY test.R ./test.R
RUN chmod +x /entrypoint.sh

CMD ["R"]