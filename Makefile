PY := conda run --no-capture-output -n lithology

.PHONY: env lock check test lint run eval data

env:            ## create the conda environment
	conda env create -f environment.yml

lock:           ## record exact resolved versions
	conda env export -n lithology --no-builds | grep -v "^prefix:" | sed "s/^      - lithology==.*/      - -e ./" > environment.lock.yml

lint:           ## static checks + formatting
	$(PY) ruff check .
	$(PY) ruff format --check .

test:           ## unit + end-to-end tests (dataset tests skip if data is absent)
	$(PY) pytest -q

check: lint test

data:           ## audit the dataset and write data/manifest.json
	$(PY) lith check-data

run:            ## process every dev image -> out/
	$(PY) lith run dataset/images --out out

eval:           ## score out/ against hand labels (dev split)
	$(PY) lith eval --pred out --labels labels --split dev
