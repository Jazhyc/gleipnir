LATEXMK ?= latexmk
JOB := report

LATEXMK_FLAGS := -pdf -jobname=$(JOB) -outdir=. -auxdir=build -emulate-aux-dir \
	-interaction=nonstopmode -halt-on-error -file-line-error -silent
# Lets pdflatex find build/preamble.fmt by name (a path in -fmt breaks -recorder).
FAST_ENGINE := -pdflatex='pdflatex -fmt=preamble %O %S'
export TEXFORMATS := ./build:

.PHONY: all pdf watch fast fmt check arxiv clean

all: pdf

# Clean, standard build: what Overleaf and arXiv will reproduce.
pdf:
	mkdir -p build
	$(LATEXMK) $(LATEXMK_FLAGS) main.tex

# Rebuild on every save; pair with a PDF viewer that reloads.
watch: build/preamble.fmt
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc -view=none \
		$(FAST_ENGINE) main.tex

# Same as pdf, but loads the precompiled preamble instead of re-reading it.
fast: build/preamble.fmt
	$(LATEXMK) $(LATEXMK_FLAGS) $(FAST_ENGINE) main.tex

fmt: build/preamble.fmt

# Dump everything in main.tex up to \endofdump into a format file.
build/preamble.fmt: main.tex preamble.tex neurips_2026.sty
	mkdir -p build
	pdflatex -ini -interaction=nonstopmode -halt-on-error -output-directory=build \
		-jobname=preamble '&pdflatex' mylatexformat.ltx main.tex > /dev/null \
		|| { tail -20 build/preamble.log; exit 1; }

MAX_MAIN_PAGES := 9

# Report the main-text page count, then list drafting markers, undefined
# references and overfull boxes.
check: pdf
	@pages=$$(sed -n 's/^\\newlabel{main-text-end}{{[^}]*}{\([0-9]*\)}.*/\1/p' build/$(JOB).aux); \
	if [ "$$pages" -gt $(MAX_MAIN_PAGES) ]; then \
		echo "Main text: $$pages pages (limit $(MAX_MAIN_PAGES)): OVER"; \
	else \
		echo "Main text: $$pages pages (limit $(MAX_MAIN_PAGES))"; \
	fi
	@grep -n '\\todo{' main.tex sections/*.tex || true
	@grep -E 'undefined|Overfull' build/$(JOB).log || true

# Source bundle for arXiv: TeX, style, bibliography output (.bbl) and assets.
arxiv: pdf
	rm -rf arxiv arxiv.tar.gz
	mkdir -p arxiv/sections
	cp main.tex preamble.tex neurips_2026.sty arxiv/
	cp sections/*.tex arxiv/sections/
	cp build/$(JOB).bbl arxiv/main.bbl
	find assets -type f \( -name '*.pdf' -o -name '*.png' -o -name '*.jpg' \) \
		-exec cp --parents {} arxiv/ \;
	tar -czf arxiv.tar.gz -C arxiv .
	@echo "Wrote arxiv.tar.gz; upload it as the source bundle."

clean:
	rm -rf -- build arxiv arxiv.tar.gz
