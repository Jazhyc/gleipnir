LATEXMK ?= latexmk

.PHONY: all pdf clean

all: pdf

pdf:
	mkdir -p build
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf presentation.pdf

clean:
	rm -rf -- build
