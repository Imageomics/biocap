lint: fmt
	ruff check train_and_eval/imageomics

fmt:
	ruff format train_and_eval/imageomics

test: lint
	pytest train_and_eval/imageomics/
