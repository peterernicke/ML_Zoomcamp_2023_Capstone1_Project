# ML Zoomcamp 2023 Capstone 1 Project - Military Aircraft Detection
|![](Images/cover-bingImageCreator.jpg)|
|:--:|
|*[Image created Microsoft Bing Image Creator](https://www.bing.com/images/create)*|

## Problem Description
The used dataset is from [Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data). It is a military aircraft detection dataset. This is a very huge dataset with more than 33.3k image files of 46 different military aircrafts. For this project I chose images from the "crop" folder which was still too much (>20.5k images). At the end I only used images from the A400M, C130, Su57, and Tu160 folder for this project. The goal is to detect the correct type of aircraft on image data. 

![](Images/examples.jpg)

## Data
As you can see on the images above, they are neither standardized in format nor in size. I expect a challenging detection process. There is another challenge, because the dataset is very imbalanced. So this will also have an impact.

![](Images/train_dataset.png)

The test dataset consists of 10 images per class. (Images are rescaled here for better visual representation).

![](Images/test_dataset.png)


This project provides a model to predict this aircraft type. 
There is a notebook (/Notebook/notebook.ipynb) that contains:
- data preparation 
- EDA
- feature importance analysis
- Training of multiple models
- Tuning of multiple models 
- Selecting the best model

The model is provided as containerized web service that listens on port 9797.

## Reproducibility
It's convenient to track my progress using the Makefile, which provides all the necessary commands:
1. To set up the environment, use "make environment" or follow the listed commands.
2. Inspect the Jupyter notebook in the Notebook folder.
3. Start model training of the final model with "make train", saving it in the Model folder.
4. Deploy the web server in a Docker container listening on port 9797 with "make deploy".
5. Test the deployment with "make test_deploy" using a sample patient record.

Additional commands for managing the environment:
- "make stop_docker" to halt running Docker containers.
- "make clean" for environment cleanup
- "make deactivate_environment" to deactivate the environment in the current terminal (in case I forget about the "deactivate" command)