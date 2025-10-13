# Install LaTeX

MiKTeX is the most popular LaTeX distribution for Windows and works seamlessly with pandoc.

https://miktex.org/download

# Install `pandoc`

```
winget install --source winget --exact --id JohnMacFarlane.Pandoc
```

# Markdown to PDF

Convert markdown file to PDF by such commands:

```
pandoc proposal.md --bibliography=references.bib --citeproc -o proposal.pdf
```
