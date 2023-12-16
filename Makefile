export PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VERBOSITY := -1

environment:
	@echo "Building Python environment"
	python3 -m pip install --upgrade pip
	pip3 install --upgrade pipenv
	pipenv install --python 3.9

train:
	@echo "Starting training..."
	pipenv run python "Script/train.py"

deploy:
	@echo "Deploying model as web service in a docker container"
	pipenv run docker build -t heart-service:v1 .
	pipenv run docker run -it --rm -p 9797:9797 heart-service:v1

test_deploy:
	pipenv run python "Script/predict-test.py"

stop_docker:
	@echo "To stop all running docker containers run"
	@echo "You need to type the next command in a terminal"
	@echo "docker stop $(docker ps -a -q)"

clean:
	@echo "Cleaning"
	pipenv --rm

deactivate_environment:
	deactivate