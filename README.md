

# Capstone Project - Machine Learning with Microsoft Azure

*This project is part of the Machine Learning Engineer with Microsoft Azure Nanodegree Project*<br>

## Project Overview 

![Project Architecture](./images/architecture.png)

In this project, two models are created for the [Winconsin Breast Cancer](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) dataset from Kaggle. One model is created using **Automated ML (AutoML)** and the second model is a custom-coded **Random Forest model** whose hyper parameters are tuned using **HyperDrive**. The performance of both the models are compared and the best model is deployed. The model can then be consumed from the generated REST endpoint. 


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
The details are documented in [eda.ipynb](./eda.ipynb).

#### 3. Encode Non-numerical Values
The column *diagnosis* was encoded using LabelEncoder from Scikit-learn as: <br>Malignant - 1 and Benign - 0. <br>

The dataframe was then saved as a Comma Separated Values (CSV) file.

### Task
The task here to classify the given details of the FNA image as malignant or benign and thus a **binary classification** algorithm is required. All the features except the *ID Number* is being used for training the model and the column *diagnosis* is considered as the taget variable. 

### Making the Dataset Accessible

Once the dataset is analysed and prepared, it is uploaded to this GitHub repository from where the raw URL of the file is obtained. The dataset is accessed from the workspace through this URL. <br>
`web_path = "https://raw.githubusercontent.com/JoanneJons/azure-machine-learning-capstone/main/breast-cancer-dataset.csv?token=AJ5V2OGXYLJ22BGYXN4EUODAC6P4K"`

## Automated ML

### Automated ML Configuration

For this project, AutoML was configured using an instance of the  `AutoMLConfig` object. The following parameters were set:<br>
`experiment_timeout_minutes = 30`<br>
*Maximum amount of time in minutes that all iterations combined can take before the experiment terminates.*<br>
For this project, this has been set as 30 because of the time restrictions of Udacity labs.<br><br>
`task = 'classification'`<br>
*The type of task to run depending on the automated ML problem to solve.*<br>
This project handles a binary classification task.<br><br>
`compute_target=cpu_cluster`<br>
*The Azure Machine Learning compute target to run the AutoML experiment on.*<br>For this experiment, a compute cluster called `cpu_cluster` is created before configuring AutoML. This computer cluser is *STANDARD_D2_V2* with a maximum of 4 nodes.
![Compute](./images/compute-automl.png)<br><br>
`training_data = train_data`<br>
*The training data to be used within the experiment.*<br>Here `train_data` is a TabularDataset loaded from a CSV file.<br><br>
`primary_metric = 'accuracy'`<br>
*The metric that AutoML will optimize for model selection.*<br><br>
`label_column_name = 'diagnosis'`<br>
*The name of the label column.*<br>Here the target column is 'diagnosis' which specifies whether the instance is malignant (1) or benign (0).<br><br>
`n_cross_validations = 5`<br>
*The number of cross validations to perform when user validation data is not specified.*<br><br>

### Automated ML Run

Submit the experiment and pass the `AutoMLConfig` instance. The `RunDetails` widget shows the the training process and details during the run. 

![rundetailswidget](./images/automl-run-details-widget.png)

After the run is completed, the Experiment tab shows the status of the run as *Completed* as seen in the screenshot below. 
![AutoML complete](./images/automl-experiment.png)

The run details page shows more information about the run, like the duration, compute target, run ID, run summary and best model summary. 
![rundetails](./images/automl-run-details.png)

In this experiment the following model models were trained by AutoML during the run. The advantage of using AutoML is that a lot of algorithms are tried out in a very short amount of time and an optimal one can be easily chosen.  
![algorithms](./images/automl-diff-models.png)

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

In this experiment, AutoML generated a model which uses the **Standard Scaler Wrapper and Logistic Regression** algorithm and has shown an accuracy of **0.97660**.
![bestmodel](./images/automl-best-model-summary.png)

![bestmodel](./images/automl-best-model.png)

### Hyperparameters Generated

**Logisitic regression** is a binary classification algorithm which uses the logistic function. This function approximates the probability of a set of binary classes (1/0). The following hyperparameters were generated for the model by AutoML:<br><br>
`C = 0.8286427728546842`<br>
**Inverse of regularization strength.**<br>
In any Machine Learning model, there is a chance of overfitting, which is a phenomenon where the model becomes 'too comfortable' with the training data that it does not generalize well. Regularization combats overfitting by making the model coefficients smaller. A larger C means less regularization and smaller C means better regularization. <br>

`class_weight = None`<br>
**Denotes the weights associated with classes.**<br>
None denotes that all classes are supposed to have weight one.<br>

`dual = False`<br>
**Dual or primal formulation.**<br>
It is not prefered to be implemented when number of samples is greater than the number of features. <br>

`fit_intercept=True` <br>
**Specifies if a constant should be added to the decision function.**<br>

`intercept_scaling=1` <br>
**Used only when the solver 'liblinear' is used, default=1.**<br>

`l1_ratio=None`<br>
**Elastic-Net mixing parameter and is only used if penalty='elasticnet'.**<br>

`max_iter=100`<br>
**Maximum number of iterations taken for the solvers to converge.**<br>

`multi_class='multinomial'`<br>
**Method of fitting labels**<br>
For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary.<br>

`n_jobs=1` <br>
**Number of CPU cores used when parallelizing over classes.**<br>

`penalty='l1'` <br>
**Used to specify the norm used in the penalization.**<br>
L1 regularization adds an L1 penalty equal to the absolute value of the magnitude of coefficients.<br>

`random_state=None`<br>
**Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data.**<br>
                                    
`solver='saga'` <br>
**Algorithm to use in the optimization problem.**<br>
'saga' is the extension of Stochastic Average Gradient descent that also allows for L1 regularization. It is a variation of gradient descent and incremental aggregated gradient approaches that uses a random sample of previous gradient values.<br>

`tol=0.0001` <br>
**Tolerance for stopping criteria.**<br>

`verbose=0`<br>
**Verbosity**<br>

`warm_start=False`<br>
**Reuse the solution of the previous call to fit as initialization.**<br>

**Standard scaler wrapper** is used to standardize and scale the values of the features. This is done because machine learning algorithms perform better when features are on a relatively smaller scale. In this dataset, different features are measured using different units and the scale varies. Thus, standardizing the values and changing the range of the values before applying logisitc regression will result in a better performance. The model used the default values of the hyperparameters, which are: <br>

`copy = True`<br>
**Create a copy for doing scaling.**<br>

`with_mean = True`<br>
**Center the data before scaling.**<br>

`with_std = True`<br>
**Scale the data to unit variance.**<br>

### Explainability

Azure Automated ML has a feature called *Explainability* which shows information about the data after the model is trained. In this experiment, the *Explainability* tab visualised the importance of each feature in the dataset. This gives ingsights about how to improve the dataset in the future and get better performances. 
![bestmodel](./images/automl-best-feature-imp.png)

### Save and Register the Best Model

The best model from the AutoML run is registered into the workspace. 
![bestmodel](./images/models-summary.png)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

### Random Forest Model
The random forest is an ensemble algorithm which can be used for both classification and regression. The algorithm creates multiple decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting. This algorithm was chosen because it is considered as highly accuracte, robust and powerful. 

The random forest model requires many hyperparameters, of the which the following parameters and their hyperparameter spaces were selected. 

`n_estimator` (int)<br>
**The number of trees in the forest.**<br>

`min_samples_split` (int)<br>
**The minimum number of samples required to split an internal node.**<br>

`max_features` {'auto', 'sqrt', 'log2'}<br>
**The number of features to consider when looking for the best split.**<br>
This function is applied on the number of features.<br>

`bootstrap` (bool)<br>
**Whether bootstrap samples are used when building trees.**<br>

### HyperDrive Configuration

Since all the hyperparameters are discrete, their space is defined as a `choice` among discrete values. <br>
`'--n_estimator': choice(100, 200, 500, 800, 100)`<br>

`'--min_samples_split': choice(2, 5, 10)`<br>

`'--max_features': choice('auto', 'sqrt', 'log2')`<br>

`'--bootstrap': choice(True, False)`<br>

The parameter sampling method used here is **random sampling**, which supports both discrete and continuous hyperparameters. It also supports early termination. The hyperparameter values are randomly selected from the defined search space. 

### Hyperdrive Run

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

### Save and Register Best Model

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
