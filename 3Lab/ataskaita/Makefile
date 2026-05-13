NAME := Sablonas
TEX := $(NAME).tex
PDF := $(NAME).pdf
AUX := out

# -lualatex            -> lualatex variklis (reikalauja luaotfload + polyglossia)
# -interaction=...     -> nestabdyti dėl menkų klaidų
# -file-line-error     -> aiškesni pranešimai
# -shell-escape        -> kai kurie paketai (chktex, listings) reikalauja
# -emulate-aux-dir + -aux-directory=$(AUX) -> visus tarpinius failus laikom out/
# Pastaba: -Werror buvo per griežtas (visi warning'ai virsta klaidomis); pašalinta.
LATEXMK_FLAGS := -lualatex \
                 -interaction=nonstopmode \
                 -file-line-error \
                 -shell-escape \
                 -emulate-aux-dir -aux-directory=$(AUX)

.PHONY: pdf
pdf:
	mkdir -p $(AUX)
	latexmk $(LATEXMK_FLAGS) -halt-on-error $(TEX)

.PHONY: watch
watch:
	mkdir -p $(AUX)
	latexmk $(LATEXMK_FLAGS) -pvc -view=none $(TEX)

# Reikalingi paketai Debian / Ubuntu / WSL Ubuntu sistemoms.
# texlive-full yra pats paprasčiausias variantas (~5 GB), tačiau ilgai diegiasi.
# Žemiau – minimalus rinkinys, kurio pakanka šiam šablonui:
#   texlive-luatex          - lualatex + luaotfload (LuaTeX šriftų varikliukas)
#   texlive-latex-extra     - tocloft, titlesec, hyperref, etc.
#   texlive-fonts-extra     - papildomi šriftai
#   texlive-lang-other      - polyglossia (lietuvių kalba)
#   texlive-bibtex-extra    - biblatex
#   biber                   - biber bibliografijos backend'as
#   texlive-science         - amsmath, amssymb, mathtools, physics
#   texlive-pictures        - graphicx, kt.
.PHONY: ubuntu
ubuntu:
	@echo "Diegiami minimalūs LaTeX paketai (LuaLaTeX + polyglossia + biber + extras)"
	sudo apt-get update
	sudo apt-get install -y \
		texlive-luatex \
		texlive-latex-extra \
		texlive-fonts-extra \
		texlive-lang-other \
		texlive-bibtex-extra \
		texlive-science \
		texlive-pictures \
		biber \
		latexmk

.PHONY: ubuntu-full
ubuntu-full:
	@echo "Diegiamas pilnas TeX Live distribucinis paketas"
	sudo apt-get update
	sudo apt-get install -y texlive-full biber latexmk

.PHONY: wordcount
wordcount:
	texcount -total -utf8 $(TEX)

.PHONY: check
check:
	chktex $(TEX) || true

.PHONY: clean
clean:
	rm -rf $(AUX)
	rm -f $(wildcard *.out *.run.xml *.log *.blg *.bbl *.aux \
		*.toc *.bcf *.synctex.gz *.fdb_latexmk *.fls *.xdv *.nav *.snm *.lof *.lot)

.PHONY: distclean
distclean: clean
	rm -f $(PDF)
