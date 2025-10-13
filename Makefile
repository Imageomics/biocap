install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-training:
	python -m pip install -r biocap_requirements.txt

lint:
	find train_and_eval/evaluation -iname '*.py' | xargs ruff check
