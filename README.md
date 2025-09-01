# Auxiliary Population-based Multitask Optimization
# The instructions

## 1.Content list

- APMTO -The code of APMTO;
- AEMTO -The reproduced code of AEMTO;
- OTMTO -The reproduced code of OTMTO;
- The random seeds for 30 runs of all the algorithms;
- data -The raw numerical output for APMTO and all the 6 compared algorithms;
- seed -The random seeds for APMTO, AEMTO, and OTMTO.
- WilcoxonRankSumTest -The code of the Wilcoxon's rank-sum test;
- ConvergenceCurve -The code to generate the convergence curves;
- README.md -The instruction file.

## 2.The acquisition of the experimental results

Herein, we have offered the code of our APMTO and the reproduced codes of AEMTO and OTMTO. The codes of the three algorithms are all organized with the same structure. By running the main.py file, the algorithms will run for 30 times on all the problems in CEC2022 test suite, and output three files:

- \<Algorithm name\>.csv -The mean values of the 30 runs of all the tasks.
- \<Algorithm name\>.txt -The results for all runs on each task, including the repeated time, the name of task, and the final result.
- \<Algorithm name\>\_fit.csv -The raw numerical output for all runs, which is a table with 30 lines (30 runs) and 20 columns (20 tasks).

We have set all the running parameter in the code of the three algorithms. The seeds for algorithms are automatically read by the program when putting it into the same directory of the code with the file name of "seed.csv".

## 3.The MToP Platform

Besides AEMTO and OTMTO, the experimental data for the rest four algorithms are acquired from the MToP platform, which can be downloaded at [intLyc/MTO-Platform: Multitask Optimization Platform (MToP): A MATLAB Benchmarking Platform for Evolutionary Multitasking](https://github.com/intLyc/MTO-Platform).

## 4.The Wilcoxon's Rank-sum Test

After acquiring all the experimental data of the 7 algorithms, the Wilcoxon's rank-sum test will be run to analysis the statistical significance. First, all the raw numerical output should be put into the directory "WilcoxonRankSumTest\source". Next, rename these files or modify the key "compares" in the config.json file to keep the file name consistent with the json file. Last, run the "Test.py" to get the result of the Wilcoxon's rank-sum test. The results will be in the directory "WilcoxonRankSumTest\output" named as "outputAPMTO.xlsx".

## 5.The Convergence Curve

