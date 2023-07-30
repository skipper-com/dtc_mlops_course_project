#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="listing-price-prediction:v1"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} --build-arg x1="${AWS_ACCESS_KEY_ID}" --build-arg x2="${AWS_SECRET_ACCESS_KEY}" --build-arg x3="${RUN_ID}" ..
    #docker build -t listing-price-prediction:v1 --build-arg x1=${AWS_ACCESS_KEY_ID} --build-arg x2=${AWS_SECRET_ACCESS_KEY} --build-arg x3=${RUN_ID} .
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker run --name price-prediction-service --rm -p 9696:9696 -d listing-price-prediction:v1

sleep 5

python3 test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker logs price-prediction-service
    docker stop price-prediction-service
    exit ${ERROR_CODE}
fi

docker stop price-prediction-service
