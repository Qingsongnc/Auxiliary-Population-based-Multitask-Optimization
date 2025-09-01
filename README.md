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

Herein, we have offered the code of our APMTO and the reproduced codes of AEMTO and OTMTO. The codes of the three algorithms are all organized with the same structure. By running the main.py file, the algorithms will run for 30 times on all the problems in CEC2022 test suite, and output three main files and a file folder with the convergence curves:

- \<Algorithm name\>.csv -The mean values of the 30 runs of all the tasks.
- \<Algorithm name\>.txt -The results for all runs on each task, including the repeated time, the name of task, and the final result.
- \<Algorithm name\>\_fit.csv -The raw numerical output for all runs, which is a table with 30 lines (30 runs) and 20 columns (20 tasks).
- Curve\<Algorithm name\> -The convergence curves for all the tasks.

We have set all the running parameter in the code of the three algorithms. The seeds for algorithms are automatically read by the program when putting it into the same directory of the code with the file name of "seed.csv".

## 3.The MToP Platform

Besides AEMTO and OTMTO, the experimental data for the rest four algorithms are acquired from the MToP platform, which can be downloaded at [intLyc/MTO-Platform: Multitask Optimization Platform (MToP): A MATLAB Benchmarking Platform for Evolutionary Multitasking](https://github.com/intLyc/MTO-Platform).

## 4.The Wilcoxon's Rank-sum Test

After acquiring all the experimental data of the 7 algorithms, the Wilcoxon's rank-sum test will be run to analysis the statistical significance. First, all the raw numerical output should be put into the directory "WilcoxonRankSumTest/source". Next, rename these files or modify the key "compares" in the "config.json" file to keep the file name consistent with the json file. Last, run the "Test.py" to get the result of the Wilcoxon's rank-sum test. The results will be in the directory "WilcoxonRankSumTest/output" named as "outputAPMTO.xlsx".

## 5.The Convergence Curve

To acquire the convergence curves for algorithms on all the tasks, we offer a code to generate the convergence curves in the directory of "ConvergenceCurve/". First, the folder convergence curves for each algorithm should be put into the directory "ConvergenceCurve/Curves". Then, check the folder names. The folder names should be consistent with the key "APMTO" in the "config.json" file. Last, run the "main.py" to generate the data for the convergence curves on all the task, which can be found in the folder "Output/APMTO".

After acquiring the data, the use of the software "origin" is necessary to generate the final convergence curves. First, import all the "csv" data in the output file into the origin. Directly copy the data in the "csv" files or use the import function offered by origin both work, but we recommend to the former one, which can be realized by click "Data -> Connect Multiple Files". Then, a necessary check is important to promise that only the column named "#FEs" is set as "X" and other columns are all set as "Y". If it is not that case, right click the column and click "set as". Next, for each imported table, select all the data and click "Plot -> Basic 2D -> Line + Symbol" to generate the final convergence curves. For the problem of the styles and layout, we do not give more description about them, please refer to the official software tutorials.
