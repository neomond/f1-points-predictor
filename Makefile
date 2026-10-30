.PHONY: install test validate train tune evaluate charts all

install:
	pip install -r requirements-dev.txt

validate:
	python src/validate.py data/f1_raw.csv

test:
	pytest -q

train:
	python src/train.py

tune:
	python src/tune.py

evaluate:
	python src/evaluate.py

charts:
	python make_post_image.py && mv linkedin_results.png feature_importance.png assets/

all: validate test train evaluate
