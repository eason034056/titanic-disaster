FROM r-base:4.3.1

# Install system dependencies (some R packages require)
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package installation script
COPY install_packages.R .

# Install R packages
RUN Rscript install_packages.R

# Copy analysis script
COPY titanic_analysis.R .

# Execute R script
CMD ["Rscript", "titanic_analysis.R"]