import os
import numpy as np
from Multi_Population import MultiPops, MPPopulation, MPIndividual, main


def binomial_crossover(v, x, CR=None):
    if CR is None:
        CR = np.random.uniform(0.1, 0.9)
    v1, x1 = v.copy(), x.copy()
    if len(v1) < len(x1):
        padd = np.zeros(len(x1) - len(v1))
        v1 = np.append(v1, padd)
    elif len(v1) > len(x1):
        x1 = x1[:len(v1)]
    jr = np.random.randint(x.size)
    rands = np.random.random(x.size)
    rands[rands > CR] = 1
    rands[rands <= CR] = 0
    U = rands * x1 + (1 - rands) * v1
    U[jr] = v[jr]
    U[U > 1] = 1
    U[U < 0] = 0
    return U


def SUS(p: np.ndarray, size):
    n = np.zeros(p.size, np.int8)
    sep = 1. / p.size
    r = np.random.random()
    k = 0
    m = 0
    for _ in range(size):
        while m < r:
            m = m + p[k]
            k = (k + 1) % p.size
        n[(k + p.size - 1) % p.size] += 1
        r = r + sep
    return n


def RWS(p):
    p = p / (p.sum() + 1e-6)
    r = np.random.random()
    m = 0
    for pi in range(p.size):
        m += p[pi]
        if r <= m:
            return pi
    return p.size - 1


class Individual(MPIndividual):
    def __init__(self, Task, X=None, **kwargs):
        super().__init__(Task, X, **kwargs)


class Population(MPPopulation):
    def __init__(self, Task, Num=100, maxDim=50, F=0.5, CR=0.9, **kwargs):
        super().__init__(Task, Individual, Num, **kwargs)
        self.CR = CR
        self.F = F
        self.qs = 0
        self.qo = 0
        self.ptsf = 0

    def Evolve(self, MPop, **kwargs):
        # V = np.zeros(shape=(self.size, self.Task.Dimension))
        # U = np.zeros(shape=(self.size, self.Task.Dimension))
        # # Generate difference vector
        # for i in range(self.size):
        #     rands = np.random.choice(range(0, self.size - 1), 3, replace=False)
        #     rands[rands >= i] += 1
        #     V[i] = self.Population[rands[0]].X + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
        #     # Binomial crossover
        # for i in range(self.size):
        #     U[i] = binomial_crossover(V[i], self.Population[i].X, self.CR)
        # # Selelction
        # r = 0
        # for i in range(self.size):
        #     fn = self.Task.function(U[i])
        #     if fn < self.Task.function(self.Population[i].X):
        #         self.Population[i].X = U[i]
        #         if fn < self.best:
        #             self.best = fn
        #         r = r + 1

        # Generate difference vector
        r = 0
        for Popi in range(self.Num):
            rands = np.random.choice(range(0, self.Num - 1), 3, replace=False)
            rands[rands >= Popi] += 1
            V = self.Population[rands[0]].X + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
            # Binomial crossover
            U = binomial_crossover(V, self.Population[Popi].X, self.CR)
            # Selelction
            Offspring = Individual(self.Task, U)
            MPop.FEIn()
            if Offspring.function < self.Population[Popi].function:
                self.Population[Popi] = Offspring
                if Offspring.function < self.best:
                    self.best = Offspring.function
                    self.gbest = Offspring.X.copy()
                r = r + 1
        return r / self.Num


class AEMTO(MultiPops):
    def __init__(self, Tasks, iteration=1000, Num=50, UseFE=True, MaxFEs=1e+5, CurveNode=None, **kwargs):
        super().__init__(Tasks, Population, iteration, Num, UseFE, MaxFEs, CurveNode, **kwargs)
        self.alpha = 0.3
        self.pmin = 0.3 / (self.T - 1)
        self.p_tsf_l = 0.05
        self.p_tsf_u = 0.7
        ptsf = (self.p_tsf_l + self.p_tsf_u) / 2
        q = np.zeros(self.T)
        p = np.zeros(self.T) + 1 / (self.T - 1)
        for ti in range(self.T):
            self.Ps[ti].p = p.copy()
            self.Ps[ti].q = q.copy()
            self.Ps[ti].p[ti] = 0
            self.Ps[ti].ptsf = ptsf
            self.Ps[ti].q[ti] = q[ti]

    def interKT(self, Tindex, **kwargs):
        # task = self.Ps[Tindex].Task
        K = []  # (a,b), a represents the index of population and b stands for individual in a
        # SUS
        n = SUS(self.Ps[Tindex].p, self.Ps[Tindex].Num)
        # RWS
        for si in range(self.T):
            if si == Tindex:
                continue
            functions = np.array([In.function for In in self.Ps[si].Population])
            # functions = np.array([self.Task[Tindex].function(In.X) for In in self.Ps[si].Population])
            for ki in range(n[si]):
                In = RWS(functions)
                K.append((si, In))
        # binomial crossover
        ns = np.zeros(self.T, np.int8)
        np.random.shuffle(K)
        for ki, k in enumerate(K):
            U = binomial_crossover(self.Ps[k[0]].Population[k[1]].X, self.Ps[Tindex].Population[ki].X)
            Offspring = Individual(self.Tasks[Tindex], U)
            self.FEIn()
            if Offspring.function < self.Ps[Tindex].Population[ki].function:
                ns[k[0]] = ns[k[0]] + 1
                self.Ps[Tindex].Population[ki] = Offspring
                if self.Ps[Tindex].Population[ki].function < self.Ps[Tindex].best:
                    self.Ps[Tindex].best = self.Ps[Tindex].Population[ki].function
                    self.Ps[Tindex].gbest = self.Ps[Tindex].Population[ki].X.copy()
        # update q, p
        for si in range(self.T):
            if si != Tindex and n[si]:
                self.Ps[Tindex].q[si] = self.alpha * self.Ps[Tindex].q[si] + (1 - self.alpha) * ns[si] / n[si]
        for si in range(self.T):
            if si != Tindex:
                self.Ps[Tindex].p[si] = self.pmin + (1 - (self.T - 1) * self.pmin) * self.Ps[Tindex].q[si] / (
                        self.Ps[Tindex].q.sum() - self.Ps[Tindex].q[Tindex] + 1e-6)
        return ns.sum() / self.Ps[Tindex].Num

    def Optimize(self, **kwargs):
        for ti in range(self.T):
            if np.random.random() <= self.Ps[ti].ptsf:
                r = self.interKT(ti)
                self.Ps[ti].qo = self.alpha * self.Ps[ti].qo + (1 - self.alpha) * r
            else:
                r = self.Ps[ti].Evolve(self)
                self.Ps[ti].qs = self.alpha * self.Ps[ti].qs + (1 - self.alpha) * r
            self.Ps[ti].ptsf = \
                self.p_tsf_l + (self.p_tsf_u - self.p_tsf_l) * self.Ps[ti].qo / (self.Ps[ti].qo + self.Ps[ti].qs + 1e-6)


# def biagen(center, b, size=50):
#     if b == 0:
#         r = np.zeros(shape=size)
#     else:
#         r = np.random.uniform(-b, b, size=size)
#     o = r + center
#     return o


# if __name__ == '__main__':
#     bias = [0, 0.005, 0.025, 0.05]
#     iteration = 1000
#     f = open('output.csv', 'w')
#     for bia in bias:
#         biavec = [
#             biagen(24, bia * 100),
#             biagen(25, bia * 100),
#             biagen(-170.9687, bia * 1000),
#             biagen(50, bia * 200),
#             biagen(25, bia * 100),
#             biagen(-26, bia * 100),
#             biagen(-25, bia * 100),
#             biagen(-670.9687, bia * 1000),
#             biagen(-50, bia * 200),
#             biagen(-25, bia * 100),
#         ]
#         tasks = [
#             Rosenbrock(bias=biavec[0]),
#             Ackley(bias=biavec[1]),
#             Schwefel(bias=biavec[2]),
#             Griewank(bias=biavec[3]),
#             Rastrigin(bias=biavec[4]),
#             Rosenbrock(bias=biavec[5]),
#             Ackley(bias=biavec[6]),
#             Schwefel(bias=biavec[7]),
#             Griewank(bias=biavec[8]),
#             Rastrigin(bias=biavec[9]),
#         ]
#         mto = AEMTO(tasks)
#         for j in range(1, iteration + 1):
#             mto.optimize()
#             if j % 50 == 0:
#                 print('iteration ' + str(j) + ':')
#                 mto.GetBest()
#         mto.GetBest(f)
#     f.close()
if __name__ == '__main__':
    # Algorithm should be replaced by the Algorithm class name.
    main(iteration=1000, Num=100, time=1, Pop=AEMTO, filename='outputtest', Problems='17',
         UseFE=False, MaxFEs=1e+5, OutputCurve=False, OutputNum=50)
    # filename = 'outputPKACP'
    # f = open(filename + '.txt', 'w', encoding='UTF-8')
    # f.close()
    # iteration = 100
    # time = 30
    # outputnum = 50
    # Num = 20
    # # Ps = [
    # #     CI_HS(),
    # #     CI_MS(),
    # #     CI_LS(),
    # #     PI_HS(),
    # #     PI_MS(),
    # #     PI_LS(),
    # #     NI_HS(),
    # #     NI_MS(),
    # #     NI_LS(),
    # #     Benchmark1(),
    # #     Benchmark2(),
    # #     Benchmark3(),
    # #     Benchmark4(),
    # #     Benchmark5(),
    # #     Benchmark6(),
    # #     Benchmark7(),
    # #     Benchmark8(),
    # #     Benchmark9(),
    # #     Benchmark10(),
    # # ]
    # # Ps = [SCP()]
    # # Ps = [MaTO19(i) for i in range(6)]
    # # Ps = [WCCI2020MaTO(i) for i in range(10)]
    # dims = [20, 30, 40, 50, 100]
    # Ps = [PKACP(dim) for dim in dims]
    # outputs = []
    # Curves = []
    # outputfits = []
    # for index, tasks in enumerate(Ps):
    #     f = open(filename + '.txt', 'a', encoding='UTF-8')
    #     f.write('Population ' + str(index + 1) + ':\n')
    #     f.close()
    #     Ntasks = len(tasks)
    #     Curves = np.zeros((Ntasks, outputnum + 1))
    #     outputt = np.zeros(Ntasks)
    #     outputfitt = [np.zeros(time) for _ in range(Ntasks)]
    #     for t in range(0, time):
    #         print('Time ' + str(t + 1) + ' for population ' + str(index + 1) + ':')
    #         P = AEMTO(tasks, Num)
    #         for Nt, task in enumerate(P.Task):
    #             print('Task ' + str(Nt), task.Info(), end=' ')
    #             Curves[Nt, 0] += P.GetBest(scout=False)[Nt]
    #         print()
    #         itrt = 0
    #         for itrt in range(1, iteration + 1):
    #             P.optimize()
    #             if iteration >= 20 and itrt % int(iteration / 20) == 0:
    #                 print('Iteration', str(itrt), ':', end=' ')
    #                 P.GetBest(scout=True)
    #             if itrt % int(iteration / outputnum) == 0:
    #                 ret = P.GetBest()
    #                 for Nt in range(len(P.Task)):
    #                     Curves[Nt, int(itrt / iteration * outputnum + 0.1)] += ret[Nt]
    #         f = open(filename + '.txt', 'a', encoding='UTF-8')
    #         f.write('Time' + str(t + 1) + ':\n')
    #         ret = P.GetBest(scout=True, fout=f)
    #         f.close()
    #         for Nt in range(Ntasks): outputfitt[Nt][t] = ret[Nt]
    #         outputt += ret
    #     outputs.append(outputt)
    #     outputfits.extend(outputfitt)
    #     Curves /= time
    #     for ti in range(Ntasks):
    #         curvefilename = 'Curve' + filename
    #         os.makedirs(curvefilename, exist_ok=True)
    #         curvefilename = os.path.join(curvefilename, 'T' + str(index) + 't' + str(ti) + '.csv')
    #         np.savetxt(curvefilename, Curves[ti].reshape(1, outputnum + 1), fmt='%.15g', delimiter=',')
    #
    # # Output results
    # outputfits = np.array(outputfits).T
    # np.savetxt(filename + '_fit.csv', outputfits, fmt='%.15g', delimiter=',')
    # for i in range(len(outputs)):
    #     for j in range(len(outputs[i])):
    #         outputs[i][j] /= time
    #         print(outputs[i][j], end=',')
    #     print()
    # fcsv = open(filename + '.csv', 'w', encoding='UTF-8')
    # for i in range(len(outputs)):
    #     for j in range(len(outputs[i])):
    #         fcsv.write(str(outputs[i][j]) + ',')
    #     fcsv.write('\n')
    # fcsv.close()
