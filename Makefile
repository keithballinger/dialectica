PYTHON?=python3.11
VENV?=venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

.PHONY: venv install run-all run-all-dry run-all-auto run-all-auto-dry run-all-batch run-all-batch-dry run-from-ideas-auto run-from-ideas-auto-dry branch-compsci-7 clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi

install: venv ## Install dependencies into venv

run-all: ## Run full flow (stops for selection)
	$(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --ask-to-continue --max-cycles 30 --name default

run-all-dry: ## Run full flow in dry-run mode
	DIALECTICA_DRY_RUN=1 $(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --ask-to-continue --max-cycles 10 --name dry-run

run-all-auto: ## Run full flow with auto-select, no pauses, high cycle limit
	$(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --auto-select --seed 42 --max-cycles 1000 --name auto

#dialectica run all --constraints constraints/llm.md --all-ideas --name llm

run-all-auto-dry: ## Dry-run auto flow
	DIALECTICA_DRY_RUN=1 $(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --auto-select --seed 42 --max-cycles 1000 --name auto-dry

run-all-batch: ## Batch all ideas (no scoring) in separate runs
	$(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --all-ideas --max-cycles 1000 --name batch

run-all-batch-dry: ## Dry-run batch all ideas
	DIALECTICA_DRY_RUN=1 $(PY) -m dialectica run all --constraints constraints/quantum_ibm_cost.md --all-ideas --max-cycles 1000 --name batch-dry

IDEAS?=runs/compsci/ideas_gpt5.md
RUNNAME?=from-ideas
run-from-ideas-auto: ## Start from IDEAS file, auto-select and draft
	$(PY) -m dialectica run all --from-ideas $(IDEAS) --auto-select --seed 42 --max-cycles 1000 --name $(RUNNAME)

run-from-ideas-auto-dry: ## Dry-run start from IDEAS file, auto-select and draft
	DIALECTICA_DRY_RUN=1 $(PY) -m dialectica run all --from-ideas $(IDEAS) --auto-select --seed 42 --max-cycles 1000 --name $(RUNNAME)-dry

branch-compsci-7: ## Branch from runs/compsci and start drafting idea #7
	$(PY) -m dialectica branch --from-run runs/compsci --idea 7 --name compsci-idea-7 --start --max-cycles 1000

clean:
	rm -rf $(VENV)
