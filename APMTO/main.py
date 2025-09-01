import argparse

import numpy as np
from Multi_Population import MPIndividual, MPPopulation, MultiPops, main
from Evolve_Operator import *
import scipy

def PCA(X, n=3):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / (X_std + 1e-6)
    cov_matrix = np.cov(X_normalized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    top_eigenvectors = eigenvectors[:, :n]
    X_projected = np.dot(X_normalized, top_eigenvectors)
    return np.array(X_projected, dtype=np.float64)

class Individual(MPIndividual):
    def __init__(self, Task, X=None, **kwargs):
        super().__init__(Task, X, **kwargs)

class FitTopfunc:
    def __init__(self, Low=0, High=1, n=50, **kwargs):
        self.Low = Low
        self.High = High
        self.Dimension = n

    def function(self, Xmap, popt, **kwargs):
        Xmap[Xmap < self.Low] = self.Low
        Xmap[Xmap > self.High] = self.High
        if len(popt) < self.Dimension:
            popt = np.append(popt, popt)
        popt = popt[:self.Dimension]
        if len(Xmap) < self.Dimension:
            Xmap = np.append(Xmap, Xmap)
        Xmap = Xmap[:self.Dimension]
        return np.sqrt(np.sum((Xmap - popt) ** 2))

class FitPop(MPPopulation):
    def __init__(self, InitialPop, Task=FitTopfunc(), Num=10, CR=0.8, F=0.5, APMaxFEs=100, presetdis=None, **kwargs):
        Task.Dimension=InitialPop[0].X.shape[0]
        super().__init__(Task, MPIndividual, Num, **kwargs)
        self.Population = [MPIndividual(Task=self.Task, X=Indi.X.copy(), **kwargs)
                           for Indi in sorted(InitialPop, key=lambda x: x.function)]
        self.F = F
        self.CR = CR
        self.gbest = min(self.Population, key=lambda Indi: Indi.function).X
        self.iteration = APMaxFEs // Num
        self.presetdis = presetdis

    def Evolve(self, popt, **kwargs):
        fail = 0
        for itrt in range(self.iteration):
            succ = 0
            for i in range(self.Num):
                rands = np.random.choice(range(0, self.Num - 1), 2, replace=False)
                rands[rands >= i] += 1
                V = self.gbest + self.F * (self.Population[rands[0]].X - self.Population[rands[1]].X)
                # Binomial crossover
                U = binomial_crossover(V, self.Population[i].X, self.CR)
                # Selelction
                offspring = Individual(self.Task, U, popt=popt)
                self.Population.append(offspring)
                # if offspring.function < self.Population[i].function:
                #     self.Population[i] = offspring
                # Update Best
                if offspring.function < self.best:
                    self.best = offspring.function
                    self.gbest = offspring.X
                    succ = 1
            self.Population = Elitist_Selection(self.Population, lambda x: 1 / x.function, self.Num)
            if succ == 0:
                fail += 1
            else:
                fail = 0
            if fail == 2:
                break
            if self.best < self.presetdis:
                break
        return self.gbest

# noinspection PyAttributeOutsideInit
class Population(MPPopulation):
    def __init__(self, Task, Num, F=0.5, CR=0.8, top=20, ptsf=None, **kwargs):
        super().__init__(Task, MPIndividual, Num, **kwargs)
        self.F = F
        self.CR = CR
        self.top = top
        self.KTPop = None
        self.ptsf = ptsf

    def Evolve(self, MPop, **kwargs):
        # Evolve process for a single population,
        # MPop is used to increase the function evaluation number with function MPop.FEIn().
        Num = len(self.Population)
        for i in range(Num):
            rands = np.random.choice(range(0, Num - 1), 3, replace=False)
            rands[rands >= i] += 1
            # V = self.Population[rands[0]].X + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
            V = self.gbest + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
            # Binomial crossover
            U = binomial_crossover(V, self.Population[i].X, self.CR)
            offspring = Individual(self.Task, U)
            self.Population.append(offspring)
            # Update best
            if offspring.function < self.best:
                self.best = offspring.function
                self.gbest = offspring.X.copy()
            MPop.FEIn()
        # Eltist Selection
        self.Population = Elitist_Selection(self.Population, lambda x: 1 / x.function, self.Num)

    def FitPop(self, MPop, Source, **kwargs):
        pd = np.sqrt(np.sum((self.Population[int(self.top / 2)].X - self.gbest) ** 2))
        Map = FitPop(InitialPop=Source.Population[:self.top], popt=self.gbest.copy(), Num=self.top, presetdis=pd, 
                n=Source.Dimension, **kwargs)
        self.KTPop = Map.Evolve(popt=self.gbest.copy(), **kwargs)
        if len(self.KTPop) < self.Dimension:
            self.KTPop = np.append(self.KTPop, self.KTPop)
        self.KTPop = self.KTPop[:self.Dimension]
        # for i in range(Map.iteration * Map.Num):
        #     MPop.FEIn()

    def Transfer(self, MPop, Source, **kwargs):
        EliteTarget = np.array([Indi.X.copy() for Indi in self.Population[:self.top]])
        Dims = np.arange(Source.Dimension)
        if len(Dims) < self.Dimension:
            Dims = np.append(Dims, np.arange(Source.Dimension))
        Dims = Dims[:self.Dimension]
        EliteSource = np.array([Indi.X.copy()[Dims] for Indi in Source.Population[:self.top]])
        ptsf = 1 - np.mean((EliteTarget - EliteSource) * (EliteTarget - EliteSource))
        if self.ptsf is not None:
            ptsf = self.ptsf
        # DD = np.cov(np.vstack([[Indi.X.copy() for Indi in self.Population], [Indi.X.copy() for Indi in Source.Population]]))
        # Dinv = np.linalg.inv(DD)
        # ptsf = np.mean(np.diagonal(scipy.spatial.distance.cdist(EliteTarget, EliteSource, 'mahalanobis', VI=Dinv)))

        # ptsf = 1 - np.mean([scipy.stats.wasserstein_distance(EliteTarget[to], EliteSource[to]) for to in range(top)])

        # ESPCA = (EliteSource - EliteSource.mean(axis=0)) / EliteSource.std(axis=0)
        # ETPCA = (EliteTarget - EliteTarget.mean(axis=0)) / EliteTarget.std(axis=0)
        # ESPCA = PCA(EliteSource)
        # ETPCA = PCA(EliteTarget)
        # ptsf = (1 - np.mean([scipy.spatial.distance.cosine(ESPCA[to], ETPCA[to]) for to in range(top)]) + 1) / 2
        # ptsf = 1 / 1 + np.e ** (np.mean([scipy.spatial.distance.cosine(ESPCA[to], ETPCA[to]) for to in range(top)]) - 1)

        # ptsf = np.array([scipy.stats.ttest_ind(EliteSource[to], EliteTarget[to]).pvalue for to in range(top)])
        # ptsf = np.mean(ptsf)
        # print(ptsf)
        # exit(1)

        if np.random.random() < ptsf:
            for i in range(self.top):
                # if MPop.g % MPop.G == 0:
                #     off = binomial_crossover(self.KTPop, EliteTarget[i], self.CR)
                # else:
                #     off = binomial_crossover(EliteTarget[i], EliteSource[i], self.CR)
                # KTPop = EliteSource[0] - np.mean(EliteSource, axis=0) + np.mean(EliteTarget, axis=0)
                # KTPop = (EliteSource[0] - np.mean(EliteSource, axis=0)) / (np.std(EliteSource, axis=0) + 1e-6) * np.std(
                #     EliteSource, axis=0) + np.mean(EliteTarget, axis=0)
                off = binomial_crossover(self.KTPop, EliteSource[i], self.CR)
                # off = binomial_crossover(EliteTarget[i], EliteSource[i], self.CR)
                off = Individual(self.Task, off)
                self.Population.append(off)
                MPop.FEIn()
            self.Population = Elitist_Selection(self.Population, lambda x: 1 / x.function, self.Num)


class Algorithm(MultiPops):
    def __init__(self, Tasks, iteration=1000, Num=50, UseFE=True, MaxFEs=1e+5, CurveNode=None, G=1, **kwargs):
        super().__init__(Tasks, Population, iteration, Num, UseFE, MaxFEs, CurveNode, **kwargs)
        self.G = G
        self.g = 0
        self.Ps[0].FitPop(self, self.Ps[1])
        self.Ps[1].FitPop(self, self.Ps[0])

    def Optimize(self, **kwargs):
        self.g = self.g + 1
        # Codes for concrete optimization steps.
        for t in range(self.T):
            self.Ps[t].Evolve(self)
            if self.g % self.G == 0:
                self.Ps[t].FitPop(self, self.Ps[1 - t])
            self.Ps[t].Transfer(self, self.Ps[1 - t])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--G', type=int, default=10)
    args.add_argument('--top', type=int, default=20)
    args.add_argument('--ptsf', type=float, default=None)
    args.add_argument('--APMaxFEs', type=int, default=100)
    args.add_argument('--Num', type=int, default=100)
    args.add_argument('--F', type=float, default=0.5)
    args.add_argument('--CR', type=float, default=0.8)
    args = args.parse_args()
    G = args.G
    top = args.top
    ptsf = args.ptsf
    APMaxFEs = args.APMaxFEs
    Num = args.Num
    F = args.F
    CR = args.CR
    # Algorithm should be replaced by the Algorithm class name.
    main(iteration=1000, Num=20, time=30, Pop=Algorithm, filename='CR/output0.7', Problems='PKACP',
         UseFE=True, MaxFEs=4000, OutputCurve=False, OutputNum=50, G=1, CR=0.7)
    # main(iteration=1000, Num=20, time=30, Pop=Algorithm, filename='outputPKACP', Problems='PKACP',
    #      UseFE=True, MaxFEs=4000, OutputCurve=False, OutputNum=50, G=1, top=5)
    # main(iterati`on=1000, Num=100, time=30, Pop=Algorithm, filename='G/outputG{}'.format(G), Problems='17+22',
    #      UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, G=G)
    # main(iteration=1000, Num=100, time=30, Pop=Algorithm, filename='top/outputtop{}'.format(top), Problems='17+22',
    #      UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, G=1, top=top)
    # main(iteration=1000, Num=100, time=30, Pop=Algorithm, filename='Ptsf/outputPtsf{}'.format(int(10 * ptsf)),
    #      Problems='17+22', UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, G=1, ptsf=ptsf)
    # main(iteration=1000, Num=100, time=30, Pop=Algorithm, filename='APMax/outputAPMaxFEs{}'.format(APMaxFEs),
    #      Problems='17+22', UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, G=1, APMaxFEs=APMaxFEs)
    # main(iteration=1000, Num=Num, time=30, Pop=Algorithm, filename='Num/outputNum{}'.format(Num),
    #      Problems='17+22', UseFE=True, MaxFEs=1e+5, OutputCurve=False, OutputNum=50, G=1, top=int(Num/5))
    # main(iteration=1000, Num=100, time=30, Pop=Algorithm, filename='F/outputF{}'.format(F), Problems='17+22',
    #      UseFE=True, MaxFEs=1e+5, OutputCurve=True, OutputNum=50, G=1, F=F)
    # main(iteration=1000, Num=100, time=30, Pop=Algorithm, filename='CR/output{}'.format(CR), Problems='17+22',
    #      UseFE=True, MaxFEs=1e+5, OutputCurve=True, OutputNum=50, G=1, CR=CR)
