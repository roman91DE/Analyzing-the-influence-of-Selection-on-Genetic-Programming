# offene fragen

## symbolic regression

dataset: 
{
    "X1" : "Relative Compactness",
    "X2" : "Surface Area",
    "X3" : "Wall Area",
    "X4" : "Roof Area",
    "X5" : "Overall Height",
    "X6" : "Orientation",
    "X7" : "Glazing Area",
    "X8" : "Glazing Area Distribution",
    "y1" : "Heating Load",
    "y2" : "Cooling Load"
}

predict just heating load?
predict both seperately?
predict both in one model?

## Err Metric

MSE and MAE

use both?
focus on one?


## Allgemeine Vorgehensweise

Single Run:
Train and test both algorithms
* capture elite performance during training
* capture performance on testing data

Experiment:
do N runs and collect results

Analysis:
primary: mann whitney u test on difference between both samples performance on testing error
