LATEXMK ?= latexmk

.PHONY: all pdf clean

all: pdf

pdf:
	mkdir -p build
	$(LATEXMK) -pdf -jobname=presentation -outdir=. -auxdir=build \
		-emulate-aux-dir -interaction=nonstopmode -halt-on-error -file-line-error main.tex

clean:
	rm -rf -- build
