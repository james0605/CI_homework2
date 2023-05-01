import numpy as np
import RBFN
import random
class GA(object):
    def __init__(self, init_G_num = 5, RBFN_K = 10) -> None:
        self.Genetics = []
        self.pool = []
        self.traindata = [dist for dist in RBFN.getTrain4d()]
        for i in range(init_G_num):
            self.Genetics.append(RBFN.RBFNet(RBFN_K))

    def cal_fitness(self, Genetic):
        sum = 0
        for (front, left, right, y) in self.traindata:
            predict = Genetic.predict([front, left, right])
            # print(predict)
            sum += (y - predict)**2
            # sum += (y - predict)
        return 1/sum

    def cal_avg_fitness(self):
        fitness = []
        for Genetic in self.Genetics:
            fitness.append(self.cal_fitness(Genetic))
        return np.mean(fitness)
    # 複製
    def reproduction(self):
        pool = []
        avg_fitness = self.cal_avg_fitness()
        fitness = [self.cal_fitness(Genetic) for Genetic in self.Genetics]
        reproduction_Num = [round(fit / avg_fitness) for fit in fitness]
        print("reproduction Num = {}".format(reproduction_Num))
        for k in range(len(reproduction_Num)):
            for i in range(reproduction_Num[k]):
                pool.append(self.Genetics[k])
        return pool
    # 交配
    def crossover(self, Genetic1, Genetic2, sigma = 0.001):
        for i in range(len(Genetic1.w)):
            w1 = Genetic1.w[i]
            w2 = Genetic2.w[i]

            w1p = w1 + sigma*(w1 - w2)
            w2p = w2 - sigma*(w2 - w1)

            Genetic1.w[i] = w1p
            Genetic2.w[i] = w2p
        return Genetic1, Genetic2
    # 突變
    def mutation(self, Genetic, s = 0.01):
        for i in range(len(Genetic.w)):
            rand = np.random.uniform(low=-1.0, high=1.0)
            Genetic.w[i] += s * rand
        return Genetic
    # 訓練
    def fit(self, epoch):
        for i in range(epoch):
            print("epoch:{}/{}".format(i, epoch))
            # 複製
            self.Genetics = self.reproduction()
            # print(self.Genetics)
            # 交配
            rand_list = np.arange(0, len(self.Genetics))
            rand_list = list(rand_list)
            rand_selet = random.sample(rand_list, k=2)
            self.crossover(self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]])
            # 突變
            self.mutation(self.Genetics[random.choice(rand_list)])

            print("loss:{}".format(self.cal_loss()))

    def cal_loss(self):
        loss = 1 / self.cal_avg_fitness()
        return loss
    
    def get_w(self, i):
        return self.Genetics[i].w

if __name__ == "__main__":
    ga = GA(10, 10)
    # print("Genetic1 w:{}".format(ga.Genetics[0].w))
    # print("Genetic2 w:{}".format(ga.Genetics[1].w))
    # ga.crossover(ga.Genetics[0], ga.Genetics[1])
    # print("Genetic1 w:{}".format(ga.Genetics[0].w))
    # print("Genetic2 w:{}".format(ga.Genetics[1].w))
    ga.fit(epoch=100)



    # print(ga.reproduction())
    RBF1 = RBFN.RBFNet(k = 5)
    # ga.cal_fitness(RBF1)
    # RBF2 = RBFN.RBFNet(k = 5)
    # RBF2 = ga.mutation(RBF1)
    # print("RBF2.w = {}".format(RBF2.w))
    # RBF3, RBF4 = ga.crossover(RBF1, RBF2)
    # print("RBF4.w = {}".format(RBF4.w))
    # print(ga.get_w(i=0))
    # print(RBFN.getTrain4d())