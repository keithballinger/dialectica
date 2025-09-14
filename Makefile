PYTHON?=python3.11
VENV?=venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

.PHONY: venv install run-all run-all-dry clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi

install: venv ## Install dependencies into venv

run-all: ## Run full flow (stops for selection)
	$(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --ask-to-continue --max-cycles 30 --name default

run-all-dry: ## Run full flow in dry-run mode
	DIALECTICA_DRY_RUN=1 $(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --ask-to-continue --max-cycles 10 --name dry-run

clean:
	rm -rf $(VENV)
