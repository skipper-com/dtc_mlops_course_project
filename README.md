# DataTalk Club MLOps Course Project 

- [DataTalk Club MLOps Course Project](#datatalk-club-mlops-course-project)
  - [Introduction](#introduction)
  - [Excuses](#execuses)
  - [Problem description](#problem-description)
    - [Dataset](#dataset)
    - [Problem](#problem)
  - [Cloud](#cloud)
  - [Batch data ingestion](#batch-data-ingestion)
  - [Data warehouse](#data-warehouse)
  - [Transformations with dbt](#transformations-with-dbt)
  - [Dashboard using Looker Studio](#dashboard-using-looker-studio)
  - [Reproducibility](#reproducibility)

# Introduction
*The main goal of the project is not to be useful but shows how good (or bad) could compile all learned technolofgies and methods in solid production environment. To be honest I choose very simple task and small dataset. Fortunately, I can come up with idea how to use results in my project.
Imagine you're going to lease a private room or apt on Airbnb, but don't know the best price for a lease. My model can predict and offer you a price based on other listing parameteres.*

## Excuses
First of all, I want to ask for excuse for my poor English.
Secondly, I sacrifice some automation for the presentation of material. I hardly tried to show and reveal detail of project alongside course topics, rather than put everything in 'black box'.
Thirdly, I deliberatly tried to omit code block from this project description and leave only explanations. I did it for clear narrative and easy reading. All code blocks or files referenced in text could be found in repo.

# Problem description
## Dataset
For the project I choose [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) Airbnb listings and metrics in NYC, NY, USA (2019). Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019. The data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions. This public dataset is part of Airbnb, and the original source can be found on this [website](http://insideairbnb.com/).
The data is a *bit* outdated and can be easily re-trained on actual data from Airbnb. 

## Problem
In this project I tried to create simple prediction model which takes most important parametres of Airbnb listing in NY and show most suitable price for listing. The project could be deployed as a service and works on a response|reply basis. During restriation of listing to lease my service (project) will predict best price for owner and hint in price field on site.

# Technologies used in projects:
- Cloud: for the project I choosed AWS cloud for time save reason. I have experience with GCP, AWS and Yandex Cloud and usually prefered GCP not accidentally. But it could take lot of time re-write all scripts for GCP.
- Model tracking: I choose MLFlow because it looks like standart in MLOps
- Orchestration: I choose Prefect as a light-weight and easy tool to orchestrate jobs and tasks. My favourite tool for orchestration is Airflow, but it much more complex and requires lots of time to deploy and start.
- Monitoring: Evidently.ai easy and powerful tool perfect suitable for project like this
- CI/CD: github actions and nothing more. Hope to have some time for experiments with Gitlab in the future. Gitlab looks even better the Github
- IaC: Terraform as a standart for IaC tools, I decided not to argue with the world:)
- AWS: export creds

# Model
The first model for baseline implemented in 01-model.py file. Baseline set by the follow steps:
1. Read dataframe (one file = all data)
2. Process features:
* select only neighbourhood, room type, number of availability days
* string type for categorical features
* drop records with price out of 95% percentile
3. Split df on train, validation and test datasets
4. Train model and calculate train MSE
5. Calculate validation MSE
6. Calculate test MSE
7. Dump model and vectors to binary file
MSE is around 70 for all train/val/test.

# Experiment tracking and model registry
Next I conducted lots of experiments with different model and hyper parametres of models to chhose the best one. On this step I use MLFlow with local DB and local artifacts storage (02-tracking.py).
1. Start local MLFlow server with ```bash mlflow server --backend-store-uri sqlite:///mlflow.db```
2. Repeat all 01-model.py file to set baseline in MLFlow
3. Re-create train/val datasets to exclude test
4. Log MLFlow for experiments with XGBoost and search space for parameters
5. Log MLFlow for experiments with different models class
6. Print results
Lowest rmse is about 70.5 with XGBoost model and best params = 
- learning_rate	0.2991109761625923
- max_depth	4
- min_child_weight 10.299631482448909
- objective	reg:linear
- reg_alpha	0.030220138602381437
- reg_lambda 0.0758045390119753
- seed	42
I think it's enough for training and model are ready to production. Now it's possible to register model, put artifactts publicly available and start service.

# Workflow Orchestration
Now I have best model with hyperparametres. Then I can create pipeline to train best model and store artifacts is S3 bucket for futher use (03-orchestration.py).
1. Re-create all training steps for best model from [Experiment tracking and model registry](experiment-tracking-and-model-registry)
2. Set new experiment to place artifacts in S3 bucket.
3. Start MLFlow server with command ```bash mlflow server --backend-store-uri=sqlite:///mlflow.db --default-artifact-root="s3://mlops-course-project/mlflow/"```
4. Start prefect server (```bash prefect server start```).
5. Run flow to train best model and put artifacts in S3, also get artifact pass and run ID for next steps.
6. Then I build a deployment which will run on schedule everyday at 23:00 using command ```bash prefect deployment build 03-orchestration.py:airbnb_flow -n airbnb --cron "0 23 * * *"```.
7. Next command will make deployment ready to start ```bash prefect deployment apply airbnb_flow-deployment.yaml```.
8. Let's start queue with command ```bash prefect agent start -q 'default'``` and deployment will started on schedule.
9. To be honest steps 6-8 I considered a bit worthless, since data is fixed, so no need to re-train model. For the real-life project I should consider prefect packed in docker container placed in ECR and run on schedule using prefect server.
10. Last step export run id stored in model folder with command ```bash export RUN_ID="b0f1202cf1da42439539dbf947081ac3"```

# Model Deployment
Easy way to deploy model is to pack all needed components inside contaner, make a web service (REST API interface) to predict function inside container and it's done.
But it looks like tip of the iceberg, because container only one building block of complex system. The hard way should include: monitoring, load balance, fault tolerance, logging, management and maybe some other useful components. But leave it for DevOps course and build just container.
Most valuable piece of previous steps is model, which is stored is S# cloud. I create a docker file and put model inside. Also I put inside container necessary (to run model) libraries and libraries for REST API.
1. Pack model and prediction inside 04-deploy.py file.
2. Create Pipfile.lock with ```bash pipenv lock```
3. Create a dockerfile including prediction file and pipfile.
4. And build container with ```bash docker build -t listing-price-prediction:v1 --build-arg x1=${AWS_ACCESS_KEY_ID} --build-arg x2=${AWS_SECRET_ACCESS_KEY} --build-arg x3=$
{RUN_ID} .```
5. Then run container ```bash docker run -it --rm -p 9696:9696  listing-price-prediction:v1```

# Model monitoring


# Best Practices


# Reproducibility