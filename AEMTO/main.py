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



if __name__ == '__main__':
    # Algorithm should be replaced by the Algorithm class name.
    main(iteration=1000, Num=100, time=30, Pop=AEMTO, filename='output', Problems='22',
         UseFE=False, MaxFEs=1e+5, OutputCurve=True, OutputNum=50)
