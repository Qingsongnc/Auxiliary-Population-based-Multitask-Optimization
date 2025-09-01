import numpy as np
from Multi_Population import MPIndividual, MPPopulation, MultiPops, main
from Evolve_Operator import binomial_crossover, Roulette_Wheel_Selection, Elitist_Selection


def OAGen(Dimension, M):
    OA = np.zeros((M, Dimension), dtype=int)
    u = int(np.log2(M) + 0.1)
    for i in range(M):
        for j in range(u):
            OA[i, int(2 ** j)] = int(np.floor(i / (2 ** (u - j))) + 0.1) % 2
            for k in range(int(2 ** j)):
                if k + int(2 ** j) >= Dimension:
                    break
                OA[i, k + int(2 ** j)] = (OA[i, int(2 ** j)] + OA[i, k]) % 2
    return OA


class ForXmgb:
    def __init__(self, Low=0, High=1, n=50):
        self.Low = Low
        self.High = High
        self.Dimension = n

    def function(self, Xmap, popt, fvecs):
        Xmap[Xmap < self.Low] = self.Low
        Xmap[Xmap > self.High] = self.High
        fvect = np.sum((Xmap - popt) ** 2, axis=1)
        return np.sqrt(np.sum((fvect - fvecs) ** 2))


class DEforXgmb(MPPopulation):
    def __init__(self, Initialpop, Task=ForXmgb(), Num=50, CR=0.9, F=0.5, iteration=10, **kwargs):
        super().__init__(Task, MPIndividual, Num, **kwargs)
        self.Population = [MPIndividual(Task=self.Task, X=Indi.X.copy(), **kwargs)
                           for Indi in sorted(Initialpop, key=lambda x: x.function)[0: Num]]
        self.F = F
        self.CR = CR
        self.bestX = min(self.Population, key=lambda Indi: Indi.function).X
        self.popt = kwargs['popt']
        self.fvecs = kwargs['fvecs']
        self.iteration = 10

    def Evolve(self, **kwargs):
        for iter in range(self.iteration):
            for i in range(self.Num):
                rands = np.random.choice(range(0, self.Num - 1), 2, replace=False)
                rands[rands >= i] += 1
                V = self.bestX + self.F * (self.Population[rands[0]].X - self.Population[rands[1]].X)
                # Binomial crossover
                U = binomial_crossover(V, self.Population[i].X, self.CR)
                # Selelction
                offspring = Individual(self.Task, U, popt=self.popt, fvecs=self.fvecs)
                self.Population.append(offspring)
                # if offspring.function < self.Population[i].function:
                #     self.Population[i] = offspring
                # Update Best
                if offspring.function < self.best:
                    self.best = offspring.function
                    self.bestX = offspring.X
            self.Population = Elitist_Selection(self.Population, lambda x: 1 / x.function, self.Num)
        return self.bestX


class Individual(MPIndividual):
    def __init__(self, Task, X=None, **kwargs):
        super().__init__(Task, X, **kwargs)


class Population(MPPopulation):
    def __init__(self, Task, Num, CR=0.6, F=0.5, **kwargs):
        super().__init__(Task, Individual, Num, **kwargs)
        self.F = F
        self.CR = CR
        self.bestX = min(self.Population, key=lambda Indi: Indi.function).X

    def Evolve(self, multiPop, **kwargs):
        for i in range(self.Num):
            rands = np.random.choice(range(0, self.Num - 1), 3, replace=False)
            rands[rands >= i] += 1
            V = self.Population[rands[0]].X + self.F * (self.Population[rands[1]].X - self.Population[rands[2]].X)
            # Binomial crossover
            U = binomial_crossover(V, self.Population[i].X, self.CR)
            # Append new
            offspring = Individual(self.Task, U)
            self.Population.append(offspring)
            # Selction
            # if offspring.function < self.Population[i].function:
            #     self.Population[i] = offspring
            # Update best
            if offspring.function < self.best:
                self.best = offspring.function
                self.bestX = offspring.X
            multiPop.FEIn()
            if multiPop.FE >= multiPop.MaxFEs:
                return
        # Elitist Selection
        self.Population = Elitist_Selection(self.Population, lambda x: 1 / x.function, self.Num)


class OTMTO(MultiPops):
    def __init__(self, Tasks, iteration=1000, Num=100, UseFE=True, MaxFEs=1e+5, CurveNode=None, nb=5, **kwargs):
        super().__init__(Tasks, Population, iteration, Num, UseFE, MaxFEs, CurveNode, **kwargs)
        self.pot = np.zeros((self.T, self.T)) + 0.5
        self.pcdt = np.zeros((self.T, self.T)) + 0.5
        self.nb = nb
        self.MaxFEs = MaxFEs
        # for T in Tasks:
        #     self.MaxFEs += T.Dimension
        # self.MaxFEs *= 1000

    def Optimize(self, **kwargs):
        for t in range(self.T):
            self.Ps[t].Evolve(self, **kwargs)
            if self.FE >= self.MaxFEs:
                return
            ts = np.random.randint(0, self.T - 1)
            if ts >= t:
                ts += 1

            # from ts to t
            if np.random.random() < self.pot[t, ts]:
                xmgb = self.CTM(ts, t)
                rot = self.OT(xmgb, t)
                self.pot[t, ts] = self.pot[t, ts] * 0.95 + rot * 0.05
                if self.FE >= self.MaxFEs:
                    return
            if np.random.random() < self.pcdt[t, ts]:
                rcdt = self.CDT(ts, t)
                self.pcdt[t, ts] = self.pcdt[t, ts] * 0.95 + rcdt * 0.05
                if self.FE >= self.MaxFEs:
                    return

    def CTM(self, ts, tt):
        ctrs = np.mean([Indi.X for Indi in self.Ps[ts].Population], axis=0)
        ctrt = np.mean([Indi.X for Indi in self.Ps[tt].Population], axis=0)
        rads = np.mean(np.sqrt(np.sum((np.array([Indi.X for Indi in self.Ps[ts].Population]) - ctrs) ** 2, axis=1)))
        radt = np.mean(np.sqrt(np.sum((np.array([Indi.X for Indi in self.Ps[tt].Population]) - ctrt) ** 2, axis=1)))
        pops = sorted(self.Ps[ts].Population, key=lambda x: x.function)[0: self.nb]
        pops = np.array([p.X for p in pops])
        popt = sorted(self.Ps[tt].Population, key=lambda x: x.function)[0: self.nb]
        popt = np.array([p.X for p in popt])
        fvecs = np.sum((self.Ps[ts].bestX - pops) ** 2, axis=1) * radt / rads
        dfx = DEforXgmb(Initialpop=self.Ps[tt].Population, Task=ForXmgb(n=self.Tasks[tt].Dimension),
                        popt=popt, fvecs=fvecs, Num=self.Ps[tt].Num // 2)
        return dfx.Evolve()

    def OT(self, Xmgb, tt):
        r = np.random.randint(0, self.Ps[tt].Num)
        Xrk = self.Ps[tt].Population[r].X
        M = int(2 ** int(np.ceil(np.log2(self.Tasks[tt].Dimension + 1)) + 0.1) + 0.1)
        OA = OAGen(self.Tasks[tt].Dimension, M)
        X = np.zeros(self.Tasks[tt].Dimension)
        Xb = np.zeros(self.Tasks[tt].Dimension)
        fbest = 1e+99
        Sfup = np.zeros((2, self.Tasks[tt].Dimension))
        Sfdown = np.zeros((2, self.Tasks[tt].Dimension))
        for i in range(M):
            X = OA[i] * Xrk + (1 - OA[i]) * Xmgb
            fn = self.Tasks[tt].function(X)
            if fn < fbest:
                fbest = fn
                Xb = X
            Sfup[1] += OA[i] * fn
            Sfup[0] += (1 - OA[i]) * fn
            Sfdown[1] += OA[i]
            Sfdown[0] += 1 - OA[i]
            self.FEIn()
            if self.FE >= self.MaxFEs:
                break
        if self.FE >= self.MaxFEs:
            fn = 1e+99
        else:
            Sf = Sfup / (Sfdown + 1e-6)
            Xp = np.zeros(self.Tasks[tt].Dimension)
            for i in range(self.Tasks[tt].Dimension):
                Xp[i] = Xrk[i] if Sf[1, i] < Sf[0, i] else Xmgb[i]
            fn = self.Tasks[tt].function(Xp)
            self.FEIn()
        if fn < fbest:
            fbest = fn
            Xot = Xp
        else:
            Xot = Xb
        IndiOT = Individual(self.Tasks[tt], X=Xot)
        if self.Ps[tt].Population[r].function > IndiOT.function:
            self.Ps[tt].Population[r] = IndiOT
            if IndiOT.function < self.Ps[tt].best:
                self.Ps[tt].best = IndiOT.function
                self.Ps[tt].bestX = IndiOT.X
            rot = 1
        else:
            rot = 0
        return rot

    def CDT(self, ts, tt):
        # from ts to tt
        # source task distribution
        ctrs = np.mean([Indi.X for Indi in self.Ps[ts].Population], axis=0)
        stds = np.std([Indi.X for Indi in self.Ps[ts].Population], axis=0)
        ctrt = np.mean([Indi.X for Indi in self.Ps[tt].Population], axis=0)
        stdt = np.std([Indi.X for Indi in self.Ps[tt].Population], axis=0)
        xcdt = np.zeros(self.Tasks[tt].Dimension)
        for i in range(self.Tasks[tt].Dimension):
            probs = np.log(stds / (stdt[i] + 1e-6)) - 1.0 / 2 + ((stdt[i] ** 2) + ((
                           ctrs - ctrt[i]) ** 2)) / (2 * (stds ** 2) + 1e-6)
            probs = 1 / (probs + 1e-6)
            index, _ = Roulette_Wheel_Selection(probs)
            xcdt[i] = np.random.normal(ctrs[index], stds[index])
        IndiCDT = Individual(self.Tasks[tt], xcdt)
        r = np.random.randint(0, self.Ps[tt].Num)
        if self.Ps[tt].Population[r].function > IndiCDT.function:
            self.Ps[tt].Population[r] = IndiCDT
            if IndiCDT.function < self.Ps[tt].best:
                self.Ps[tt].best = IndiCDT.function
                self.Ps[tt].bestX = IndiCDT.X
            rcdt = 1
        else:
            rcdt = 0
        self.FEIn()
        return rcdt

if __name__ == '__main__':
    # Algorithm should be replaced by the Algorithm class name.
    main(iteration=1000, Num=50, time=30, Pop=OTMTO, filename='output', Problems='22',
         UseFE=True, MaxFEs=1e+5, OutputCurve=True, OutputNum=50)