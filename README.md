# Titanic Disaster Analysis with Docker

This project analyzes Titanic survival prediction data using both Python and R, containerized with Docker for reproducible analysis.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Download](#data-download)
3. [Project Structure](#project-structure)
4. [Running Python Analysis](#running-python-analysis)
5. [Running R Analysis](#running-r-analysis)
6. [Expected Output](#expected-output)
7. [Troubleshooting](#troubleshooting)

---


## Data Download

### Download from Kaggle

1. **Visit the Kaggle Competition Page**
   - Go to: https://www.kaggle.com/c/titanic/data
   - Log in to your Kaggle account (register if you don't have one)

2. **Download the Dataset**
   - Click the "Download All" button
   - The downloaded file will be named `titanic.zip`

3. **Extract and Place the Data**
   ```bash
   # Extract the files
   unzip titanic.zip -d titanic-data
   
   # Copy the data to the project's data directory
   cp titanic-data/*.csv src/data/
   ```

### Verify Data

Confirm that the `src/data/` directory contains the following files:
```bash
ls -la src/data/
```

You should see:
```
train.csv
test.csv
gender_submission.csv
```

---

## Project Structure

```
titanic-disaster/
├── README.md                          # This file
├── src/
│   ├── data/                          # Data directory
│   │   ├── train.csv                  # Training data
│   │   ├── test.csv                   # Testing data
│   │   ├── gender_submission.csv      # Submission example
│   │   ├── submission_python.csv      # Python prediction output (generated)
│   │   └── submission_r.csv           # R prediction output (generated)
│   ├── python/                        # Python analysis
│   │   ├── Dockerfile                 # Python Docker configuration
│   │   ├── requirements.txt           # Python package dependencies
│   │   └── titanic_analysis.py        # Python analysis script
│   └── r/                             # R analysis
│       ├── Dockerfile                 # R Docker configuration
│       ├── install_packages.R         # R package installation script
│       └── titanic_analysis.R         # R analysis script
```

---

## Running Python Analysis

### Step-by-Step Instructions

Here are the complete steps to run the Python analysis using Docker:

#### Step 1: Navigate to Python Directory

```bash
cd src/python
```

#### Step 2: Build Docker Image

```bash
docker build -t titanic-python .
```


#### Step 3: Run Docker Container

```bash
docker run -v "$(pwd)/../data:/app/data" titanic-python
```

#### Output Files

After running the Python analysis, the prediction results will be saved to:

```
src/data/submission_python.csv
```

This file contains:
- `PassengerId`: The ID of each passenger
- `Survived`: The predicted survival outcome (0 = Did not survive, 1 = Survived)

---

## Running R Analysis

### Step-by-Step Instructions

Here are the complete steps to run the R analysis using Docker:

#### Step 1: Navigate to R Directory

```bash
cd src/r
```

If you just finished running the Python analysis, you need to switch from `src/python` to `src/r`:

```bash
cd ../r
```

#### Step 2: Build Docker Image

```bash
docker build -t titanic-r .
```



#### Step 3: Run Docker Container

```bash
docker run -v "$(pwd)/../data:/app/data" titanic-r
```

#### Output Files

After running the R analysis, the prediction results will be saved to:

```
src/data/submission_r.csv
```

This file contains:
- `PassengerId`: The ID of each passenger
- `Survived`: The predicted survival outcome (0 = Did not survive, 1 = Survived)

---

## Summary

### Python Analysis (Three Commands)
```bash
cd src/python
docker build -t titanic-python .
docker run -v "$(pwd)/../data:/app/data" titanic-python
```

### R Analysis (Three Commands)
```bash
cd src/r
docker build -t titanic-r .
docker run -v "$(pwd)/../data:/app/data" titanic-r
```


