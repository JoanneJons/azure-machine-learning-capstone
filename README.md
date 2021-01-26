

# Capstone Project - Machine Learning with Microsoft Azure

*This project is part of the Machine Learning Engineer with Microsoft Azure Nanodegree Project*<br>
In this project, two models are created for the [Winconsin Breast Cancer](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) dataset from Kaggle. One model is created using Automated ML (AutoML) and the the second model is a custom-coded Random Forest model whose hyper parameters are tuned using HyperDrive. The performance of both the models are compared and the best model is deployed and consumed. 


## Project Set Up and Installation

### Create a Workspace
An Azure workspace is a container that includes data and configuration information. An Azure subscription is required to create a workspace. There are different ways to create a workspace.<br>
[Documentation - Create and manage Azure Machine Learning workspaces](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python#connect-to-a-workspace)

### Setup Docker for Running Swagger on Localhost (Optional)
[Get Docker](https://docs.docker.com/get-docker/)

## Dataset

### Overview
For this project, the dataset chosen is the **[Winconsin Breast Cancer](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)** dataset from Kaggle. There are a total of 32 columns which includes,<br>
Attribute information:
<ul>
    <li>ID Number</li>
    <li>Diagnosis (M = malignant, B = benign)</li>
</ul>
and real-valued features computed from digitized image of a Fine Needle Aspirate of a breast mass and describes the characteristics of the cell nuclei present in the image,
<ul>
    <li>radius (mean of distances from center to points on the perimeter)</li>
    <li>texture (standard deviation of gray-scale values)</li>
    <li>perimeter</li>
    <li>area</li>
    <li>smoothness (local variation in radius lengths) </li>
    <li>compactness (perimeter^2 / area - 1.0) </li>
    <li>concavity (severity of concave portions of the contour) </li>
    <li>concave points (number of concave portions of the contour)</li>
    <li>symmetry</li>
    <li>fractal dimension ("coastline approximation" - 1)</li>
</ul>
The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.

### Exploratory Data Analysis
The dataset is analysed in this project. The following steps were down for data analysis: <br>
#### 1. Check for Null Values: 
There were no null or missing values in the dataset. <br>
#### 2. Visualise Features and Target:<br>
![Malignant and Benign](./images/bandm.png)
<br>
All the features were visualised using histograms and box plots.<br>
The details are documented in eda.ipynb

#### 3. Encode Non-numerical Values
The column *diagnosis* was encoded using LabelEncoder from Scikit-learn as: <br>Malignant - 1 and Benign - 0. <br>

The dataframe was then saved as a Comma Separated Values (CSV) file.

### Task
The task here to classify the given details of the FNA image as malignant or benign and thus a binary classification algorithm is required. All the features except the *ID Number* is being used for training the model and the column *diagnosis* is considered as the taget variable. 

### Making the Dataset Accessible

Once the dataset is analysed and prepared, it is uploaded to this GitHub repository from where the raw URL of the file is obtained. The dataset is accessed from the workspace through this URL. <br>
`web_path = "https://raw.githubusercontent.com/JoanneJons/azure-machine-learning-capstone/main/breast-cancer-dataset.csv?token=AJ5V2OGXYLJ22BGYXN4EUODAC6P4K"`

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

For this project, AutoML was configured using an instance of the  `AutoMLConfig` object. The following parameters were set:<br>
`experiment_timeout_minutes = 30`<br>
*Maximum amount of time in minutes that all iterations combined can take before the experiment terminates.*<br>
For this project, this has been set as 30 because of the time restrictions of Udacity labs.<br><br>
`task = 'classification'`<br>
*The type of task to run depending on the automated ML problem to solve.*<br>
This project handles a binary classification task.<br><br>
`compute_target=cpu_cluster`<br>
*The Azure Machine Learning compute target to run the AutoML experiment on.*<br>For this experiment, a compute cluster called `cpu_cluster` is created before configuring AutoML. This computer cluser is *STANDARD_D2_V2* with a maximum of 4 nodes.<br><br>
`training_data = train_data`<br>
*The training data to be used within the experiment.*<br>Here `train_data` is a TabularDataset loaded from a CSV file.<br><br>
`primary_metric = 'accuracy'`<br>
*The metric that AutoML will optimize for model selection.*<br><br>
`label_column_name = 'diagnosis'`<br>
*The name of the label column.*<br>Here the target column is 'diagnosis' which specifies whether the instance is malignant (1) or benign (0).<br><br>
`n_cross_validations = 5`<br>
*The number of cross validations to perform when user validation data is not specified.*<br><br>

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
