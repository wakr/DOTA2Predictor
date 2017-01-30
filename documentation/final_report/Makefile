show: malli.pdf
	kpdf malli.pdf &

malli.pdf: malli.tex
	pdflatex malli

toc:
	pdflatex malli
	pdflatex malli

bib:
	pdflatex malli
	bibtex malli
	pdflatex malli
	pdflatex malli

clean:

	rm -f *.aux *.log *~ *.toc *.pdf *.out *.gz *.log *.bbl *.blg
