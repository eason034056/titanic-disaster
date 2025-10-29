options(repos = c(CRAN = "https://cloud.r-project.org"))

# List of required packages
packages <- c(
  "readr",        # For reading CSV files
  "dplyr",        # For data processing
  "tidyr",        # For data cleaning
  "caret",        # For machine learning
  "ggplot2"       # For data visualization (optional)
)

# Install packages (if not already installed)
for (package in packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Display installed package versions
cat("\nInstalled package versions:\n")
for (package in packages) {
  version <- packageVersion(package)
  cat(sprintf("%-15s : %s\n", package, version))
}