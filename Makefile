#LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
#LOCAL_IMAGE_NAME:=price-prediction:${LOCAL_TAG}

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: #quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} --build-arg x1="${AWS_ACCESS_KEY_ID}" --build-arg x2="${AWS_SECRET_ACCESS_KEY}" --build-arg x3="${RUN_ID}" .

integration_test:
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integraton-test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install
