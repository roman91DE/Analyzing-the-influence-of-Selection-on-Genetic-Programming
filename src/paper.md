---
title: |
  <center>Analyzing the influence of Selection on Genetic Programming's Generalization ability in Symbolic Regression: </center>
  <center>
    <small>
    A comparison of $\epsilon$-Lexicase Selection and Tournament Selection
     </small> 
  </center>
  <center>--------------------------------------------------------</center>
  <center>Student: Roman Höhn</center>
  <center>Date of Birth: 1991-04-14</center>
  <center>Place of Birth: Wiesbaden, Hesse</center>
  <center>Student ID: 2712497</center>
  <center>Supervisor: David Wittenberg</center>
  <center>--------------------------------------------------------</center>
  <center>Research Project: Seminar Information Systems (03.996.3299)</center>
  <center>Research Area: Computational Intelligence</center>
  <center>Chair of Business Administration and Computer Science</center>
  <center>Johannes Gutenberg University Mainz</center>
  <center>Summerterm 2022</center>
  |
date: "Date of Submission: 2022-06-21"
output:
  pdf_document:
    extra_dependencies: ["float"]
    fig_caption: yes
    number_sections: true
    includes:
      in_header: settings.tex
fontsize: 12pt
#toc: yes
# toc_depth: 4
bibliography: bib/bib/ref.bib
csl: bib/csl/harvard1.csl
#include-before: '`\newpage{} `{=latex}'
include-before: '`\thispagestyle{empty} `{=latex}'
---





\tableofcontents
\thispagestyle{empty}
\newpage

# Introduction

```{=html}
<!-- 
Structure:
  - Genetic Programming
  - Selection
  - Symbolic Regression
  - Generalization
  - Research Question
-->
```

<!-- GP --->
Genetic programming (GP), a subfield of evolutionary computation (EC), is a metaheuristic that is used to search for computer programs that solve a given problem by simulating the process of darwinian evolution. The basic principle of GP is to gradually evolve solutions by repeatedly selecting parent solutions from a randomized population of computer programs based on a fitness metric. Then, genetic operators are applied on the selected solutions to genereate new offspring candidate solutions. By repeating this process over many generations, GP acts as a guided search for high fitness solutions throughout the decision space.
A unique feature of GP among other evolutionary optimization procedures is the possibility to evolve solutions of variable length.

<!-- Selection --->
The overall performance of GP can depend strongly on the choice of its underlying operators, one crucial component in this is the operator for parent selection. 
Tournament Selection is a commonly used selection operator in EC and is the most used operator in GP systems [@10.1007/978-3-642-16493-4_19, p.181]. A parent solution is selected by randomly sampling $k$ individuals from the current population into a tournament pool and then the solution with the highest fitness score from the tournament pool is selected [@10.1007/978-3-642-16493-4_19, p.182]. 
Lexicase selection has been suggested as an alternative to tournament selection that is not based on aggregating fitness scores. It samples $n$ test cases in random order and then eliminates solutions from the selection pool on a per test case basis if they are not performing on an elite level [@6920034, p.1].

Since regular Lexicase selection has been shown to perform suboptimal on continous-valued optimization problems, a modified variation called $\epsilon$-Lexicase selection has been suggested for symbolic regression by La Cava et. al (2016). Here, $\epsilon$-Lexicase selection has shown itself to outperform both in overall performance while showing only negligible computational overhead [@epsilon_lexicase_main, p.747].

<!-- Symbolic Regression --->
The goal of symbolic regression is to find a mathematical model for an observed set of datapoints [@10.1007/978-3-540-24621-3_22, p.794]. Symbolic Regression has been one of the first GP applications and to this day is an actively studied and highly relevant area of research [@poli08:fieldguide, p.114]. In most symbolic regression problems, little to no a priori knowledge about the optimal form and structure of the target function is available. The ability of GP to optimize for model structure as well as for parameters has lead to it being one of the most prevalent methods used in the domain of symbolic regression [@10.1007/978-3-540-24621-3_22, p.795].

<!-- Generalization --->
An important quality of all supervised machine learning applications, including GP, is the ability to not only optimize performance for the test cases a model is trained on but to also perform well on previously unseen cases, this is refered to as generalization. In most real world applications of symbolic regression only a small subset of labeled data is available for training. The aim is to produce a model that not only accurately predicts the provided training data but can also predict previously unseen cases with high precision [@Gonalves2016AnEO, p.6]. A model that is extensivley optimized on the provided training data, may overfit to this data sample which may lead to a decrease in generalization.

<!-- Research Question --->
This research project tries to answer the question if the usage of $\epsilon$-Lexicase selection influences the generalization behaviour of programs that are evolved using GP for symbolic regression if compared to programs that are evolved using traditional tournament selection.


# An Overview of the current state of research

## Selection

In the first description of lexicase^[The alternative, more descriptive name given by @lexicase_first_desciption: global pool, uniform random sequence, elitist lexicase parent selection] selection for GP by @lexicase_first_desciption it was suggested as a novel parent selection method for target problems that are modal in nature. The author classified modal problems as "problems that qualitatively different modes of response are required for inputs from different regions of the problem's domain"[@lexicase_first_desciption, p.1].

In a following article Lexicase selection has been proposed specifically for the purpose of solving so called uncompromising problems with GP. @6920034 defined uncompromising problems as problems that require the final solution to perform optimal on each of the cases it is tested on, examples include symbolic regression, the design of digital multipliers or finding terms in finite algebras. The authors also provided evidence that lexicase selection can significantly improves GP's ability to solve some uncompromising problems if compared to selection methods that are based on aggregated fitness [@6920034, p.12].

Both articles argued that many of the problem domains that GP is commonly used for are prevalent for problems that are modal/ uncompromising in nature. In comparison to other selection operators, populations that are evolved using variations of the lexicase selection operator show a very high degree of genetic diversity which might be a key contributer to the improved performance [@6920034, p.1] [@epsilon_lexicase_main, p.745]. In theory, Lexicase based selection should be more prone to select solutions that are specialist (high fitness on a small subset of training cases) than solutions that are generalist (high average fitness among all training cases).


The basic algorithm for lexicase parent selection as described by @lexicase_first_desciption is given below:

```
Lexicase - Parent-Selection:

1. Initialize:
  (a) Set candidates to be the entire population.
  (b) Set cases to be a list of all of the fitness cases in random order.

2. Loop:
  (a) Set candidates to be the subset of the current candidates that have 
      exactly the best fitness of any individual currently in candidates for 
      the first case in cases.
  (b) If candidates or cases contains just a single element then return
      the first individual in candidates.
  (c) Otherwise remove the first case from cases and go to Loop.
```

More recent research by @epsilon_lexicase_main suggested $\epsilon$-Lexicase selection as a modified selection operator that can improve overall GP performance if applied to continous-valued symbolic regression tasks in comparison to tournament and standard lexicase selection.  @epsilon_lexicase_main argued that the original concept behind lexicase selection as formulated by @lexicase_first_desciption does not fit the requirements of real-world symbolic regression tasks. The authors identified the pass condition of regular lexicase selection as the main problem if applied to symbolic regression: Individual solutions can only be selected if they perform on an elite level but with noisey and continous-valued data it is very unlikely for two individuals to achieve an exactly equal error on a test case, this finally results in filtering out too many individuals during the selection process[@epsilon_lexicase_main, p.742]. $\epsilon$-Lexicase adresses this problem by introducing an $\epsilon$ parameter that specifies a range around the elite error, individuals that perform inside this range pass the current selection iteration. @https://doi.org/10.48550/arxiv.1709.05394 describe $\epsilon$-Lexicase selection as a more "relaxed version of lexicase selection".
Different methods to configure the $\epsilon$ Parameter have been explored, for the limited scope of this project I will focus on the most promising implementation of $\epsilon$-Lexicase selection that is automatically adjusted on the basis of the median absolute deviation of errors inside the selection pool [@epsilon_lexicase_main, p.742] [@https://doi.org/10.48550/arxiv.1709.05394, p.6].

The performance increase of $\epsilon$-Lexicase selection for symbolic regression problems has been demonstrated and reported for many benchmark problems which led to widespread adaption of it in symbolic regression applications [@epsilon_lexicase_main, p.744-745].

In comparison, traditional GP selection methods such as tournament selection most commonly compute the fitness of a program as the mean of its error for each individual fitness case. One downside to this approach is that the total amount of information is reduced from a wide range of individual errors to a single metric. @6920034 suspected that this loss in overall information provided to GP might reduce overall performance especially if applied to the class of uncompromising problems that require the solution to perform well on a wide range of diverse cases. Since tournament selection selects individuals based on their average fitness across all test cases, the resulting solutions should be expected to be more biased towards being generalists instead of specialists.

The basic algorithm for tournament is given below for comparison [@10.1007/978-3-642-16493-4_19, p.182-183]: 

```
Tournament - Parent-Selection:

k = tournament size

1. Sample:
  (a) Randomly sample k individuals from the current population
      into a tournament pool.
2. Select:
  (a) Compute the mean fitness for each individual inside the tournament pool
      based on all fitness cases.
  (b) Select individual from the tournament pool with the highest mean fitness.
```


## Generalization 

Generalization, the ability of a model to perform well on previously unseen cases, is one of the fundamental goals in most real world machine learning applications. @open_issues_gp raised awareness to the fact, that the topic of generalization in GP has not gotten the attention that other machine learning-based areas, e.g, deep learning, have attributed to it. A similiar statement was proposed by @generalisation_in_gp in his review of research in generalization in GP. Both authors adressed the need for additional research regarding the topic.

@generalisation_in_gp critized that almost all applications published in the initial GP literature by John Koza [@koza_main] are not using separate training and testing datasets which might result in overfitting and a poor overall generalization ability of the programs that are produced. The author calls for a more widespread adoption of methods like generational sampling of new training cases or the overall separation into training and testing datasets to improve generalization in GP [@generalisation_in_gp, p.10].

An aspect that is suspected to negatively correlate with GP's generalization is overall size and in particular bloating of the resulting programs. Bloating can be described as a growth in the total size of a program that does not also improve it's performance in any meaningful way [@bloat_overfitting_gp, p. 1].
It has been widely suspected that GP configurations that produce very large programs also have a higher tendency to specialize on difficult or unusual test cases which could result in lower generalization [@wang_wagner_rondinelli_2019, p. 268]. The minimum description length principle (MDLP) describes this phenomenom by claiming that less complex models are more likely to perform better at generalizing than more complex models that achieve a comparable fitness during the training phase [@open_issues_gp, p. 349]. However MDLP should be considered carefully, experiments by @bloat_overfitting_gp on the prediction of bioavailability for drugs demonstrated that GP based techniques can either bloat without overfitting or overfit without bloating [@bloat_overfitting_gp, p. 8].

## Symbolic Regression

The task of finding a mathematical function that fits a given set of datapoints has been one of the first applications of GP in [@koza_main, p. 238].
Practical examples given by @koza_main included tasks such as the discovery of scientific laws, finding solutions to differential and integral equations or the programmatic compression of images.

Since it's inception, GP based symbolic regression has been widely used in practical application ranging over a wide field of disciplines. A well known example of an industrial use-case has been published by @soft_sensor_gp. The authors used GP based symbolic regression to derive nonlinear functions that could be used to significantly increase the robustness of industrial sensors which resulted in wide-spread adoption and a significant reduction in costs.

Many variations of GP based symbolic regression have been proposed to improve overall performance, @wang_wagner_rondinelli_2019 summarized the main characteristics and differences for 6 common variations including procedures such as geometric semantic genetic programming, cartesian genetic programming or GP-based relevance vector machines. A large benchmarking study on symbolic regression published by @Orzechowski_2018 compared 4 GP-based procedures to several state-of-the-art machine leaning techniques. The authors find that GP, eventhough more time consuming, can achieve better results if compared to the state-of-the-art machine learning algorithm gradient-boosting.

# Experimental study

## Research Design

<!-- Dataset and Task --->
To study the influence of selection on generalization I selected a dataset about energy efficiency in buildings that is part of the UC Irvine Machine Learning Repository [@Dua:2019]. The dataset contains eight individual building attributes that map to two different outcomes, heating and cooling load, for N=768 cases [@fe8fa39e88a040bbacba5a465c48043f].

Using two otherwise identical GP systems, one deploying tournament selection and the other $\epsilon$-lexicase selection, the objective is to find a computer program that best predicts the outcome variable heating load $(Y1)$^[The symbolic regression is performed on one of the two provided outcome variables, the variable Cooling Load will be excluded.] of the buildings using a subset of the eight building attributes $(X1,..,X8)$ for input. The specific meaning of all attributes inside the dataset are described in table 1.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Overview - Energy Heating Dataset</caption>
 <thead>
  <tr>
   <th style="text-align:left;font-weight: bold;"> Variable </th>
   <th style="text-align:left;font-weight: bold;"> Description </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> X1 </td>
   <td style="text-align:left;"> Relative Compactness </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X2 </td>
   <td style="text-align:left;"> Surface Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X3 </td>
   <td style="text-align:left;"> Wall Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X4 </td>
   <td style="text-align:left;"> Roof Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X5 </td>
   <td style="text-align:left;"> Overall Height </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X6 </td>
   <td style="text-align:left;"> Orientation </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X7 </td>
   <td style="text-align:left;"> Glazing Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X8 </td>
   <td style="text-align:left;"> Glazing Area Distribution </td>
  </tr>
  <tr>
   <td style="text-align:left;"> y1 </td>
   <td style="text-align:left;"> Heating Load </td>
  </tr>
  <tr>
   <td style="text-align:left;"> y2 </td>
   <td style="text-align:left;"> Cooling Load </td>
  </tr>
</tbody>
</table>

<!-- Method:single run --->
To measure the generalization ability of each model the dataset will be randomly split in half, resulting in a training and testing dataset each containing 384 individual cases. Each model will be evolved by traditional GP using only the fitness cases present in the training dataset. For each generation the highest fitness model of the current population will be tested with the previously unseen fitness cases that are part of the testing dataset. For each run of the experiment statistics will be collected on fitness which will form the basis of my further statistical analysis. Additional statistics will be collected for the average length of the population and the length of the elite program for each generation to explore differences in their distribution and possible correlations to generalization. 

<!-- Statistical Analysis --->
Since GP is a stochastic optimization algorithm, the basic experiment will be run for a total of 50 times to ensure a fair and meaningful comparison based on a large number of runs for both algorithms. For each run of the experiment the dataset will be randomly split in half as described, both models are then trained and tested using the exact same set of fitness cases.

The statistical analysis of the collected data will first focus on examining the question if, on average, the usage of $\epsilon$-Lexicase selection will result in models that perform significantly different than models that are evolved using tournament selection both.

> $H0_{1}$: The distribution underlying the samples of training/testing errors produced by torunament selection is the same as the distribution underlying samples of training/testing errors produced by $\epsilon$-lexicase selection

The next question of interest is, to examine if statistical significant differences between the mean errors of training and testing data exist for both selection operators:

> $H0_{2}$: The distribution underlying the samples of training errors produced by $\epsilon$-Lexicase/torunament selection is the same as the distribution underlying samples of testing errors produced by $\epsilon$-Lexicase/tournament selection

To gather further insight into the differences between both GP systems, an additional test will be performed on the question if differences in the total size of the resulting programs exist:

> $H0_{3}$: Size differences exist between the distribution underlying the samples produced by $\epsilon$-Lexicase and the distribution underlying samples of tournament selection

All hypothesis will be tested using a level of significance of $\alpha=0.05$.

To further examine the difference in generalization behaviour, the mean testing and training errors over each generation will be visualized for both algorithms. The specific aim of this visualization is to explore the mechanism of overfitting and to examine if differences between both methods can be detected.

All GP experiments will be implemented by using the python programming language in conjunction with DEAP, a framework for distributed evolutionary algorithms that implements various tools and algorithms for genetic programming [@DEAP_JMLR2012].

## Gentetic Programming Configuration

### Evolutionary parameters

The basic evolutionary parameters for both systems are presented in table 2. Tournament selection is used with a default tournament size of $3$ individuals while the $\epsilon$ parameter in lexicase selection is selected automatically as previously mentioned in subsection 3.1.


<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Evolutionary Parameters</caption>
 <thead>
  <tr>
   <th style="text-align:left;font-weight: bold;"> Parameter </th>
   <th style="text-align:left;font-weight: bold;"> Value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Population Size </td>
   <td style="text-align:left;"> 500 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Number of Generations </td>
   <td style="text-align:left;"> 100 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Mutation Rate </td>
   <td style="text-align:left;"> 20% </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Crossover Rate </td>
   <td style="text-align:left;"> 80% </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Tournament Size </td>
   <td style="text-align:left;"> 3 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Epsilon selection </td>
   <td style="text-align:left;"> automatic </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Elite Size </td>
   <td style="text-align:left;"> 0 </td>
  </tr>
</tbody>
</table>

### Fitness Evaluation

The fitness $f$ for each model will be based on the mean squared error (MSE) over all fitness cases for prediction and the empirically measured values as described by eq.1.

\begin{equation}
\tag{eq. 1}
MSE = \frac{1}{n} * \sum_{i=1}^{n} (Y_i - \hat{Y_1})^2
\end{equation}

where:

-   $n$: total number of test cases
-   $Y_i$: empirical value for case $i$
-   $\hat{Y_1}$: predicted value for case $i$.

The resulting fitness function $f$ for an individual program $i$ is descibed by eq 2:

\begin{equation}
\tag{eq. 2}
 f(i, \tau ) = \frac{1}{N} * \sum_{t \epsilon \tau} (y_t - y\hat{}_t(i, x_t))^2 
\end{equation}

where^[Naming and notation was adopted from @epsilon_lexicase_main]:

-   $\tau$: the set of $N$ fitness cases
-   $y_t$: empirical value of the target for case $t$
-   $y\hat{}_t(i, x_t)$: predicted value for the target for case $t$ by running the program $i$ with the total set of input variables $x_t$


### Primitive Set

The primitve set consists of the terminals that are listed in table 3 and the functions that are listed in table 4. To avoid runtime errors I implemented protected version of the operators for division, natural logarithm and square root [@koza_main, p.82-83]. As suggested by @koza_main I also included ephemeral constants to the set of terminals to provide the evolutionay search the opportunity to explore and include randomly generated constants. 


<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Terminals</caption>
 <thead>
  <tr>
   <th style="text-align:left;font-weight: bold;"> Terminal </th>
   <th style="text-align:left;font-weight: bold;"> Description </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> X1 </td>
   <td style="text-align:left;"> Relative Compactness </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X2 </td>
   <td style="text-align:left;"> Surface Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X3 </td>
   <td style="text-align:left;"> Wall Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X4 </td>
   <td style="text-align:left;"> Roof Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X5 </td>
   <td style="text-align:left;"> Overall Height </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X6 </td>
   <td style="text-align:left;"> Orientation </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X7 </td>
   <td style="text-align:left;"> Glazing Area </td>
  </tr>
  <tr>
   <td style="text-align:left;"> X8 </td>
   <td style="text-align:left;"> Glazing Area Distribution </td>
  </tr>
  <tr>
   <td style="text-align:left;"> random_int </td>
   <td style="text-align:left;"> Ephemeral Constant (integer) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> random_float </td>
   <td style="text-align:left;"> Ephemeral Constant(float) </td>
  </tr>
</tbody>
</table>


<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Functions</caption>
 <thead>
  <tr>
   <th style="text-align:left;font-weight: bold;"> Function </th>
   <th style="text-align:right;font-weight: bold;"> Arity </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Addition </td>
   <td style="text-align:right;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Subtraction </td>
   <td style="text-align:right;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Multiplication </td>
   <td style="text-align:right;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Negation </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Sine </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Cosine </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Protected Division </td>
   <td style="text-align:right;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Protected Natural Logarithm </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Protected Square Root </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table>

### Genetic Operators

GP operators are represented in table 5. Both genetic operators use a static limit to control for the height of the reulting trees [@koza_main, p.104]. Individual programs are initialized by using the ramped half-and-half method, 50% of the population are created by using the Growth algorithm and the remaining 50% are created by using the Full algorithm [@koza_main, p.93]. 


<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>GP Operators</caption>
 <thead>
  <tr>
   <th style="text-align:left;font-weight: bold;"> Operator </th>
   <th style="text-align:left;font-weight: bold;"> Implementation </th>
   <th style="text-align:right;font-weight: bold;"> Static.Height.Limit </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Initilization </td>
   <td style="text-align:left;"> Ramped Half/Half </td>
   <td style="text-align:right;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Crossover </td>
   <td style="text-align:left;"> One Point Crossover </td>
   <td style="text-align:right;"> 17 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Mutation </td>
   <td style="text-align:left;"> Uniform Mutation </td>
   <td style="text-align:right;"> 17 </td>
  </tr>
</tbody>
</table>

The crossover operator implemented by DEAP randomly selects a crossover point in each individual and exchanges each subtree with the point as root between each individual [@DEAP_JMLR2012]. Mutation also randomly selects a point in the tree individual, it then replaces the subtree below that point as a root by the expression generated using the full grow initialization method [@DEAP_JMLR2012]. 


# Results

## Descriptive Statistics

Tables 6 and 7 summarize the results for all fitness scores collected over 50 total runs of the experiment^[All floating point numbers in text have been rounded to 3 decimal places].



\begin{table}[!h]

\caption{\label{tab:unnamed-chunk-6}Summary - Tournament Selection}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}[t]{lrrrrrrrrrr}
\toprule
\textbf{X} & \textbf{gen} & \textbf{nevals} & \textbf{mean\_training\_error} & \textbf{std\_training\_error} & \textbf{min\_training\_error} & \textbf{max\_training\_error} & \textbf{testing\_error} & \textbf{std\_testing\_error} & \textbf{avg\_size} & \textbf{elite\_size}\\
\midrule
count & 101.0 & 101.000 & 1.010000e+02 & 1.010000e+02 & 101.000 & 1.010000e+02 & 100.000 & 100.000 & 100.000 & 100.000\\
mean & 50.0 & 420.742 & 2.092614e+18 & 3.305357e+20 & 16.483 & 1.046309e+21 & 16.905 & 28.858 & 55.674 & 62.070\\
std & 29.3 & 8.115 & 2.098028e+19 & 3.313954e+21 & 12.514 & 1.049016e+22 & 11.117 & 15.732 & 34.504 & 36.850\\
min & 0.0 & 417.300 & 2.582030e+08 & 1.463390e+10 & 7.066 & 1.193720e+11 & 7.847 & 16.217 & 3.331 & 4.340\\
25\% & 25.0 & 418.780 & 2.151894e+10 & 3.044239e+12 & 8.479 & 1.070232e+13 & 9.350 & 18.344 & 25.133 & 28.195\\
\addlinespace
50\% & 50.0 & 420.000 & 3.494635e+11 & 5.511534e+13 & 11.918 & 1.747285e+14 & 12.874 & 22.715 & 57.299 & 66.400\\
75\% & 75.0 & 421.000 & 4.892936e+12 & 6.455802e+14 & 18.773 & 2.238973e+15 & 19.685 & 32.750 & 87.852 & 96.625\\
max & 100.0 & 500.000 & 2.108540e+20 & 3.330558e+22 & 77.243 & 1.054272e+23 & 62.693 & 95.282 & 109.761 & 115.340\\
\bottomrule
\end{tabular}}
\end{table}



\begin{table}[!h]

\caption{\label{tab:unnamed-chunk-7}Summary - Epsilon-Lexicase Selection}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}[t]{lrrrrrrrrrr}
\toprule
\textbf{X} & \textbf{gen} & \textbf{nevals} & \textbf{mean\_training\_error} & \textbf{std\_training\_error} & \textbf{min\_training\_error} & \textbf{max\_training\_error} & \textbf{testing\_error} & \textbf{std\_testing\_error} & \textbf{avg\_size} & \textbf{elite\_size}\\
\midrule
count & 101.0 & 101.000 & 1.010000e+02 & 1.010000e+02 & 101.000 & 1.010000e+02 & 100.000 & 100.000 & 100.000 & 100.000\\
mean & 50.0 & 420.781 & 7.533781e+12 & 1.186094e+15 & 22.013 & 3.762638e+15 & 22.456 & 39.048 & 58.416 & 63.819\\
std & 29.3 & 8.109 & 5.900311e+13 & 9.313627e+15 & 16.670 & 2.950203e+16 & 15.904 & 19.225 & 34.515 & 36.356\\
min & 0.0 & 416.240 & 4.660413e+05 & 3.314323e+07 & 9.680 & 1.539125e+08 & 11.129 & 21.487 & 3.219 & 4.280\\
25\% & 25.0 & 418.900 & 2.528193e+08 & 2.727451e+10 & 11.316 & 1.192373e+11 & 12.089 & 24.860 & 28.233 & 29.615\\
\addlinespace
50\% & 50.0 & 420.040 & 2.196352e+09 & 2.647194e+11 & 14.478 & 1.098068e+12 & 15.223 & 30.057 & 64.689 & 71.440\\
75\% & 75.0 & 421.020 & 2.027020e+11 & 3.198174e+13 & 23.511 & 1.012976e+14 & 23.305 & 51.081 & 87.394 & 94.910\\
max & 100.0 & 500.000 & 5.788911e+14 & 9.137151e+16 & 74.622 & 2.894465e+17 & 77.518 & 110.244 & 108.785 & 115.660\\
\bottomrule
\end{tabular}}
\end{table}


<!-- Statistical Analysis 

For both experiments the best of the run models that are evolved using $\epsilon$-Lexicase Selection on average perform better as measured by their mean error score. This behaviour is consistent during both training and testing phases of the experiment  

$\epsilon$-Lexicase based GP achieves an average minimum fitness of 3.166/1.576 (MSE/MAE) during model training and a score of 2.949/1.678 during the testing phase on previously unseen data. Suprisingly the models that are trained using MSE as a metric for fitness evaluation, on average, perform even better on unseen cases as they do on the initial training data.

In comparison, the models that are trained using tournament selection perform with an average minimum fitness of 10.9887/2.32173 (MSE/MAE) during model training and 11.146/2.196 during the testing phase. Again we observe an improved performance in testing error, this time models that are evolved using MAE for fitness perform better on testing data than on their initial training data.

The initial results seem support the two alternative hypothesis of this research project:

- $HA_{testing}$: T $\epsilon$-Lexicase Selection produces models that differ in their generalization behaviour if compared to models that are evolved using torunament selection
  
- $H0_{training}$: $\epsilon$-Lexicase Selection produces models that differ in their performance during model training if compared to models that are evolved using torunament selection

--->

## Inferential Statistics 

Figure 1 visualizes the distribution of fitness scores achieved by both GP systems after finishing 100 full generations for all 50 runs of the experiment. 

![Distribution of Errors](./plots/mean_error_boxplot_all.png)

... interpretation...

The four samples from figure 1 have been tested for normal distribution by computing the D’Agostino and Pearson test [@DAgostino1971AnOT] [@DAgostino1973TestsFD] to determine if they satisfy the requirements of the students t-test. The results are summarized in table 8. Since not all samples test positive for a normal distribution a Mann-Whitney U (MWU) ranksum test will be performed to test for statistical significance. 


\begin{table}[!h]

\caption{\label{tab:unnamed-chunk-8}Normal Distribution Tests}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}[t]{lrrrl}
\toprule
\textbf{sample} & \textbf{statistic} & \textbf{p.value} & \textbf{alpha} & \textbf{normal\_distributed}\\
\midrule
Tournament - Training Errors & 36.146 & 0.000 & 0.05 & False\\
E-Lexicase - Training Errors & 1.575 & 0.455 & 0.05 & True\\
Tournament - Testing Errors & 32.503 & 0.000 & 0.05 & False\\
E-Lexicase - Testing Errors & 59.135 & 0.000 & 0.05 & False\\
\bottomrule
\end{tabular}}
\end{table}

The results of the MWU test are summarized in table 9.


\begin{table}[!h]

\caption{\label{tab:unnamed-chunk-9}Mean Error - P-Values}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}[t]{lrrrr}
\toprule
\textbf{ } & \textbf{tournament\_elite\_size} & \textbf{elexicase\_elite\_size} & \textbf{tournament\_avg\_size} & \textbf{elexicase\_avg\_size}\\
\midrule
tournament\_elite\_size & 1.000 & 0.858 & 0.533 & 0.652\\
elexicase\_elite\_size & 0.858 & 1.000 & 0.764 & 0.682\\
tournament\_avg\_size & 0.533 & 0.764 & 1.000 & 0.992\\
elexicase\_avg\_size & 0.652 & 0.682 & 0.992 & 1.000\\
\bottomrule
\end{tabular}}
\end{table}

... interpretation...

To further examine the generalization behaviour based on both selection operators I plotted the mean fitness scores for both algorithms in figure 2. 

![Mean Errors](./plots/mean_error_combined.png)

... interpretation...

In a final step I analyzed the growth behaviour for both GP systems. Figure 3 visualizes the average size of individuals inside the population as well as the average size of the best performing individual for each generation of the evolution.

![Mean Size](./plots/size_combined.png)
... interpretation...

Again, a MWU test is conducted to test for statistical differences in the underlying distribution of the sizes measured during the experiment. The results are summarized in table 10.


\begin{table}[!h]

\caption{\label{tab:unnamed-chunk-10}Mean Size - P-Values}
\centering
\resizebox{\linewidth}{!}{
\begin{tabular}[t]{lrrrr}
\toprule
\textbf{ } & \textbf{tournament\_elite\_size} & \textbf{elexicase\_elite\_size} & \textbf{tournament\_avg\_size} & \textbf{elexicase\_avg\_size}\\
\midrule
tournament\_elite\_size & 1.000 & 0.858 & 0.533 & 0.652\\
elexicase\_elite\_size & 0.858 & 1.000 & 0.764 & 0.682\\
tournament\_avg\_size & 0.533 & 0.764 & 1.000 & 0.992\\
elexicase\_avg\_size & 0.652 & 0.682 & 0.992 & 1.000\\
\bottomrule
\end{tabular}}
\end{table}




# Conclusion

# Limitations and open questions

I faced two major difficulties in the preparation and evaluation of this research project:

  1. Configuration of evolutionary parameters
  2. Limited computational ressources
  
The combination of both issues might have resulted in a significant reduction in overall robustness of the results represented in sections 5 and 6. The performance of all evolutionary algorithms, including GP, can be highly dependant on a large number of parameters and implementation details. Although the series of experiments conducted in this project are comparably simple in nature, the results achieved might still be highly influenced by the GP configuration detailed in subsection 4.2. 

To gain further insight into the differences in generalizaion behaviour and to adress the research question with a higher level of certainty, the experiment I conducted should be repeated for many different GP configurations. Some exemplary parameters that could drastically influence the results of this experiment include the population size, the total number of generations, the inclusion of elitism or the computation of $\epsilon$ in lexicase selection. Other factors whose influence on generalization behaviour could be important include different methods to compute an individuals fitness (e.g. mean absolute error or root mean squared error) and testing out different variations of the genetic operators for mutation and crossover. 

Another obvious weakness of my project is that the experiment is only based on a single symbolic regression application, the prediction of the heating load of buildings based on a relativley small dataset. To gain further trust in the obtained results, the experiment should be repeated for other symbolic regression tasks and datasets. At the current state it remains unknown if the generalization behaviour of both selection operators might be highly problem specific and dependant on the dataset used for training and testing.

All limitations and shortcomings described above are also connected to the second limitation of this research project, the limited amount of computational ressources. The total computation time consumed to run the current experiments (50 runs, 2 algorithms, 100 generations, 500 individuals) has already been close to 24 hours on a modern, arm-based Apple-M1 system. Eventhough the overall computation time could probably be reduced by further optimization of the source code, e.g. using tools for parallel computation, the ressources necessary to repeat the whole experiment for different parameter configurations would certainly be beyond the scope of this research project. 




\newpage  

# I   References {.unnumbered #I}

::: {#refs}
:::
\newpage


# II  Statutory Declaration {.unnumbered #II}

