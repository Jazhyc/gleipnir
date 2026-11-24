LATEXMK ?= latexmk
JOB := proposal

LATEXMK_FLAGS := -pdf -jobname=$(JOB) -outdir=. -auxdir=build -emulate-aux-dir \
	-interaction=nonstopmode -halt-on-error -file-line-error -silent

.PHONY: all pdf watch check clean

all: pdf

pdf:
	mkdir -p build
	$(LATEXMK) $(LATEXMK_FLAGS) main.tex

# Rebuild on every save; pair with a PDF viewer that reloads.
watch:
	mkdir -p build
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc -view=none main.tex

# Page count, then drafting markers, undefined references and overfull boxes.
check: pdf
	@echo "Pages: $$(pdfinfo $(JOB).pdf | sed -n 's/^Pages: *//p')"
	@grep -n '\\todo{' main.tex sections/*.tex || true
	@grep -E 'undefined|Overfull' build/$(JOB).log || true

clean:
	rm -rf -- build
