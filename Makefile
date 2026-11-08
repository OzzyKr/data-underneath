PY := conda run --no-capture-output -n lithology

.PHONY: env lock check test lint run eval data dash-data dash-ui dash dash-demo

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

dash-data:      ## dashboard data + scan tiles from out/ (holdout logs hidden)
	$(PY) lith dash-build

dash-ui:        ## build the dashboard front end
	cd dashboard && npm install --no-audit --no-fund && npm run build

dash: dash-ui   ## serve the dashboard (dev logs) at http://127.0.0.1:8000
	$(PY) lith dash --open

dash-demo: dash-ui  ## all logs including holdout, then serve and open the browser
	$(PY) lith dash-build --all
	$(PY) lith dash --open
