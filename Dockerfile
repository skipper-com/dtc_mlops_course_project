FROM python:3.10.6-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

RUN pip3 --no-cache-dir install --upgrade awscli

ARG x1
ARG x2
ARG x3

ENV AWS_ACCESS_KEY_ID $x1
ENV AWS_SECRET_ACCESS_KEY $x2

COPY [ "04-deploy.py", "./" ]

RUN echo $x1

RUN aws s3 cp s3://mlops-course-project/mlflow/2/$x3/artifacts/models_pickle/lin_reg.bin .
RUN aws s3 cp s3://mlops-course-project/mlflow/2/$x3/artifacts/preprocessor/preprocessor.b .

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "04-deploy:app" ]
