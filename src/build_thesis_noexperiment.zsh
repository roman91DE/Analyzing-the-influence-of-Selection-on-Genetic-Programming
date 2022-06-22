#!/bin/zsh

# This script is used to create all statistical evaluation and render the pdf output document

# activate the conda environment that is specified in src/gp_research.yml
conda activate gp_research

# execute the notebook that creates tables and plots
ipython ./statistics.ipynb   

# process rmarkdown document to pdf output
Rscript -e "rmarkdown::render('../docs/rmd/paper.Rmd',  encoding = 'UTF-8')"
