# Install LaTeX

MiKTeX is the most popular LaTeX distribution for Windows and works seamlessly with pandoc.

https://miktex.org/download

# Install `pandoc`

```
winget install --source winget --exact --id JohnMacFarlane.Pandoc
```

# Citation style

Different citation styles can be downloaded as `.csl` files from:

https://www.zotero.org/styles 

# Markdown to PDF

Finally, markdown file can be converted to PDF by a command like:

```
pandoc proposal.md --bibliography=references.bib --citeproc --csl=ieee.csl -o proposal.pdf --filter pandoc-crossref
```

```
pandoc "pre-proposal draft notes.md" --bibliography=references.bib --citeproc --csl=ieee.csl -o "pre-proposal draft notes.pdf"
```
