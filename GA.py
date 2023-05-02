import numpy as np
import RBFN
import random
class GeneticOpt(object):
    def __init__(self, init_G_num = 5, RBFN_K = 10, dump = False) -> None:
        self.Genetics = []
        self.pool = []
        self.traindata = [dist for dist in RBFN.getTrain4d()]
        self.dump = dump
        self.is_done = False
        for i in range(init_G_num):
            self.Genetics.append(RBFN.RBFNet(RBFN_K))

    def cal_fitness(self, Genetic):
        sum = 0
        for (front, left, right, y) in self.traindata:
            predict = Genetic.predict([front, left, right])
            # print(predict)
            sum += abs(y - predict)**2
            # sum += (y - predict)
        return 1/(sum)

    def cal_avg_fitness(self):
        fitness = []
        for Genetic in self.Genetics:
            fitness.append(self.cal_fitness(Genetic))
        return np.mean(fitness)
    
    def check_Genetic_w(self):
        print("Genetic Pool :")
        for i, Genertic in enumerate(self.Genetics):
            print("Genertic {} weight:{}".format(i, Genertic.w))

    # 複製
    def reproduction(self):
        pool = []
        avg_fitness = self.cal_avg_fitness()
        fitness = [self.cal_fitness(Genetic) for Genetic in self.Genetics]
        print([0 if (fit / avg_fitness) < 0 else fit / avg_fitness for fit in fitness])
        self.reproduction_Num = [0 if round(fit / avg_fitness) < 0 else round(fit / avg_fitness) for fit in fitness]
        print("reproduction Num = {}".format(self.reproduction_Num))
        for k in range(len(self.reproduction_Num)):
            for i in range(self.reproduction_Num[k]):
                pool.append(self.Genetics[k])
        return pool
    # 交配
    def crossover(self, Genetic1, Genetic2, sigma = 0.3):
        for i in range(len(Genetic1.w)):
            w1 = Genetic1.w[i]
            w2 = Genetic2.w[i]

            w1p = w1 + sigma*(w1 - w2)
            w2p = w2 - sigma*(w2 - w1)

            Genetic1.w[i] = w1p
            Genetic2.w[i] = w2p
        return Genetic1, Genetic2
    # 突變
    def mutation(self, Genetic, s = 0.3):
        for i in range(len(Genetic.w)):
            rand = np.random.uniform(low=-1.0, high=1.0)
            Genetic.w[i] += s * rand
        return Genetic
    # 訓練
    def fit(self, epoch, early_stop = False):
        count = 0
        early_stop_count = 5
        for i in range(epoch):
            print("epoch:{}/{}".format(i, epoch))
            if self.dump:
                print("{:-^50s}".format("Origin"))
                self.check_Genetic_w()
            # 複製
            self.Genetics = self.reproduction()
            if self.dump:
                print("{:-^50s}".format("After Reproduction"))
                self.check_Genetic_w()
            # print(self.Genetics)
            # 交配
            rand_list = np.arange(0, len(self.Genetics))
            rand_list = list(rand_list)
            rand_selet = random.sample(rand_list, k=2)
            self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]] =  self.crossover(self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]])
            if self.dump:
                print("{:-^50s}".format("After Crossover"))
                print("Choose {} and {}".format(rand_selet[0], rand_selet[1]))
                self.check_Genetic_w()
            # 突變
            muta_rand = random.choice(rand_list)
            self.Genetics[muta_rand] = self.mutation(self.Genetics[muta_rand])
            if self.dump:
                print("{:-^50s}".format("After Mutation"))
                print("Choose {}".format(muta_rand))
                self.check_Genetic_w()
            print("loss:{}".format(self.cal_loss()))

            self.flag = False
            
            if early_stop:
                for i in self.reproduction_Num:
                    if i == 1:
                        self.flag=True
                    else:
                        self.flag=False
                        count = 0
                        break
                if count == early_stop_count:
                    break
                count += 1
        return random.choice(self.Genetics)

    def cal_loss(self):
        loss = 1 / self.cal_avg_fitness()
        return loss
    
    def get_w(self, i):
        return self.Genetics[i].w
    
    def predict(self, state):
        for i, Genetic in enumerate(self.Genetics):
            print("Genetic {} predict {}".format(i, Genetic.predict(state)))

if __name__ == "__main__":
    ga = GeneticOpt(10, 10)
    # print("Genetic1 w:{}".format(ga.Genetics[0].w))
    # print("Genetic2 w:{}".format(ga.Genetics[1].w))
    # ga.crossover(ga.Genetics[0], ga.Genetics[1])
    # print("Genetic1 w:{}".format(ga.Genetics[0].w))
    # print("Genetic2 w:{}".format(ga.Genetics[1].w))
    ga.reproduction()
    
    # ga.fit(epoch=50)
    # ga.predict([9.7355, 10.9379, 18.5740])


    # print(ga.reproduction())
    # RBF1 = RBFN.RBFNet(k = 5)
    # ga.cal_fitness(RBF1)
    # RBF2 = RBFN.RBFNet(k = 5)
    # RBF2 = ga.mutation(RBF1)
    # print("RBF2.w = {}".format(RBF2.w))
    # RBF3, RBF4 = ga.crossover(RBF1, RBF2)
    # print("RBF4.w = {}".format(RBF4.w))
    # print(ga.get_w(i=0))
    # print(RBFN.getTrain4d())