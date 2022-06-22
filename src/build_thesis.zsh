#!/bin/zsh

# This script is used to run the basic GP experiment for N_RUN iterations, create all statistical evaluation and render the pdf output document

N_RUN=50

# activate the conda environment that is specified in src/gp_research.yml
conda init zsh
conda activate gp_research

# run the basic experiment for N_RUN times
for IND in {1..$N_RUN}
    do
        echo "Starting $IND of $N_RUN run:"
        python ./main.py $IND
    done

# execute the notebook that creates tables and plots
ipython ./statistics.ipynb   

# process rmarkdown document to pdf output
Rscript -e "rmarkdown::render('../docs/rmd/paper.Rmd',  encoding = 'UTF-8')"
