import json
import os
import numpy as np

def MergeTask(c, t, Algs, output):
    filename = 'T' + str(c) + 't' + str(t) + '.csv'
    OutPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Output', output)
    Outfile = os.path.join(OutPath, filename)
    datafile = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Curves')
    data = []
    for alg in Algs:
        dataForAlg = os.path.join(datafile, alg, filename)
        algdata = np.loadtxt(dataForAlg, delimiter=',')
        data.append(algdata)
    FEs = range(0, 102, 2)
    data.insert(0, FEs)
    data = np.array(data, dtype=np.float64)
    data[data < 0] = 0
    os.makedirs(OutPath, exist_ok=True)
    titles = Algs.copy()
    titles.insert(0, 'FEs')
    np.savetxt(Outfile, data.T, fmt='%.15g', delimiter=',', header=",".join(titles, ))


if __name__ == '__main__':
    config = json.load(open('config.json'))
    Algorithm = config['algorithmset']
    outputfile = config['output']
    Algorithms = config[Algorithm]
    CaseNum = config['CaseNum']
    TaskNum = config['TaskNum']
    for i in range(CaseNum):
        for j in range(TaskNum):
            MergeTask(i, j, Algorithms, outputfile)