.PHONY: install test lint benchmark benchmark-all train template-survey clean

# =======================================================================
# GSR-CACL Makefile
# Convenient aliases for common commands.
# Usage: make <target> [VAR=value]
# =======================================================================

# Default values
DATASET ?= finqa
MODE    ?= gsr
EPOCHS  ?= 5
SAMPLE  ?= 300

# -----------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------
install:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[dev,gpu]"

# -----------------------------------------------------------------------
# Template coverage survey (Reviewer 4: "template coverage chưa khảo sát")
# -----------------------------------------------------------------------
template-survey:
	python -m gsr_cacl.template_coverage_analysis \
		--dataset $(DATASET) --sample $(SAMPLE)

template-survey-all:
	python -m gsr_cacl.template_coverage_analysis --dataset finqa --sample 300
	python -m gsr_cacl.template_coverage_analysis --dataset tatqa --sample 300
	python -m gsr_cacl.template_coverage_analysis --dataset convfinqa --sample 300

# -----------------------------------------------------------------------
# Retrieval benchmark
# -----------------------------------------------------------------------
benchmark:
	python -m gsr_cacl.benchmark_gsr \
		--mode $(MODE) --dataset $(DATASET) --top-k 3 \
		--embedding intfloat/multilingual-e5-large-instruct

benchmark-all:
	@echo "=== FinQA ==="
	python -m gsr_cacl.benchmark_gsr --mode gsr --dataset finqa
	python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset finqa
	@echo "=== TAT-DQA ==="
	python -m gsr_cacl.benchmark_gsr --mode gsr --dataset tatqa
	python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset tatqa
	@echo "=== ConvFinQA ==="
	python -m gsr_cacl.benchmark_gsr --mode gsr --dataset convfinqa
	python -m gsr_cacl.benchmark_gsr --mode hybridgsr --dataset convfinqa

# -----------------------------------------------------------------------
# Compare with baselines
# -----------------------------------------------------------------------
compare:
	python -m gsr_cacl.evaluate_comparison \
		--dataset $(DATASET) --methods gsr hybridgsr \
		--save outputs/gsr_benchmark/$(DATASET)/comparison.csv

# -----------------------------------------------------------------------
# Training (3 stages)
# -----------------------------------------------------------------------
train-identity:
	python -m gsr_cacl.train --dataset $(DATASET) \
		--stage identity --epochs 3 --lr 1e-4

train-structural:
	python -m gsr_cacl.train --dataset $(DATASET) \
		--stage structural --epochs 3 --lr 1e-4

train-joint:
	python -m gsr_cacl.train --dataset $(DATASET) \
		--stage joint --epochs $(EPOCHS) --batch-size 16 --lr 5e-5 \
		--margin 0.2 --lambda 0.5

train-all-stages:
	$(MAKE) train-identity DATASET=$(DATASET)
	$(MAKE) train-structural DATASET=$(DATASET)
	$(MAKE) train-joint DATASET=$(DATASET) EPOCHS=$(EPOCHS)

# -----------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------
lint:
	ruff check src/gsr_cacl/
	ruff format src/gsr_cacl/

test:
	pytest src/gsr_cacl/tests/ -v

test-kg:
	python -c "from gsr_cacl.kg import build_kg_from_markdown, ConstraintKG; \
		table = '''| Revenue | COGS | Gross Profit | \
		|---|---|---| \
		| 100 | 60 | 40 |'''; \
		kg = build_kg_from_markdown(table); \
		print(f'Nodes: {len(kg.nodes)}, Edges: {len(kg.edges)}'); \
		print(f'Template: {kg.template.name if kg.template else None}'); \
		assert len(kg.nodes) > 0; \
		print('KG builder OK')"

test-chap:
	python -c "from gsr_cacl.kg import build_kg_from_markdown; \
		from gsr_cacl.negative_sampler import CHAPNegativeSampler; \
		table = '''| Revenue | COGS | Gross Profit | \
		|---|---|---| \
		| 100 | 60 | 40 |'''; \
		kg = build_kg_from_markdown(table); \
		sampler = CHAPNegativeSampler(); \
		negs = sampler.sample(kg, n_negatives=3); \
		print(f'Generated {len(negs)} CHAP negatives'); \
		for n in negs: print(f'  {n.perturbation_type}: {n.original_value} -> {n.new_value}'); \
		assert len(negs) >= 1; \
		print('CHAP sampler OK')"

test-templates:
	python -c "from gsr_cacl.templates import TEMPLATES, match_template, _normalize_header; \
		headers = ['Revenue', 'COGS', 'Gross Profit', 'Operating Expenses', \
		           'Operating Income', 'Income Tax', 'Net Income']; \
		tpl, conf = match_template(headers); \
		print(f'Template: {tpl.name if tpl else None}, Confidence: {conf:.3f}'); \
		assert tpl is not None; \
		print(f'Constraints: {len(tpl.constraints)}'); \
		print('Template library OK')"

# -----------------------------------------------------------------------
# Ablation study (§4.2 in paper)
# -----------------------------------------------------------------------
ablation:
	python -m gsr_cacl.benchmark_gsr --mode gsr --dataset $(DATASET) \
		--ablation gsr-kg   # GSR without KG
	python -m gsr_cacl.benchmark_gsr --mode gsr --dataset $(DATASET) \
		--ablation gsr-constraint  # GSR without constraint score
	python -m gsr_cacl.benchmark_gsr --mode gsr --dataset $(DATASET) \
		--ablation cacl-chap  # CACL with random negatives

# -----------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------
clean:
	rm -rf outputs/
	rm -rf src/gsr_cacl.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
