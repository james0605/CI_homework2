import numpy as np
import RBFN
import random
class GeneticOpt(object):
    def __init__(self, G_num = 5, RBFN_K = 10, dump = False) -> None:
        self.Genetics = []
        self.pool = []
        self.traindata = [dist for dist in RBFN.getTrain4d()]
        self.dump = dump
        self.is_done = False
        self.REPET_fra= 0.3
        self.GEN_NUM = 15
        for i in range(G_num):
            self.Genetics.append(RBFN.RBFNet(RBFN_K))
    
    def cal_fitness(self, Genetic):
        sum = 0
        for (front, left, right, y) in self.traindata:
            predict = Genetic.predict([front, left, right])
            # print(predict)
            # predict = float(predict*80.0-40.0)
            sum += abs(y - predict)
            # sum += (y - predict)
        return 1/sum
    
    def cal_fitness_v2(self):
        apha = 0.2
        self.objective_values = [self.cal_fitness(Genetic) for Genetic in self.Genetics]
        min_obj_value = min(self.objective_values)
        max_obj_value = max(self.objective_values)
        fitness = []
        for value in self.objective_values:
            fitness.append(max(apha*(max_obj_value-min_obj_value), 10**-5) + (max_obj_value - value))
        avg_fitness = np.mean(fitness)
        return fitness, avg_fitness


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
        
        # fitness, avg_fitness = self.cal_fitness_v2()

        # 線性調整
        # min_fit = min(fitness)
        # max_fit = max(fitness)
        # fitness = np.array(fitness)
        # fitness = fitness - min_fit
        # d = max_fit - min_fit
        # if d == 0:
        #     d=1
        # print(fitness)
        # fitness /= d
        # print(type(fitness))
        # avg_fitness = np.mean(fitness)
        # print("avg_fitness :{}".format(avg_fitness))
        pool = []
        

        fitness = np.array([self.RMSE_loss(Genetic) for Genetic in self.Genetics])
        sorted_fitness = np.argsort(fitness)
        print("BEST Genetic RMSE loss :{}".format(self.RMSE_loss(self.Genetics[sorted_fitness[0]])))
        REPEAT_NUM = int(self.REPET_fra * self.GEN_NUM)+1
        if len(sorted_fitness) < self.GEN_NUM:
            for i in range(self.GEN_NUM-len(sorted_fitness) + 1):
                sorted_fitness.append(0)
        else :
            sorted_fitness = sorted_fitness[:self.GEN_NUM]
        print("sorted_fitness : {}".format(sorted_fitness))
        for i in range(REPEAT_NUM):
            pool.append(self.Genetics[sorted_fitness[0]])
            
        for i in range(1, self.GEN_NUM-REPEAT_NUM+1):
            pool.append(self.Genetics[sorted_fitness[i]])
            
        print("POOL NUM : {}".format(len(pool)))
        print("sorted fitness arg:{}".format(sorted_fitness))
        # avg_fitness = self.cal_avg_fitness()
        # reproduction_Num = fitness/avg_fitness
        # print("reproduction Num = {}".format(reproduction_Num))
        # reproduction_Num = np.rint(reproduction_Num).astype(int)

        # print("reproduction Num = {}".format(reproduction_Num))
        # for k in range(len(reproduction_Num)):
        #     for i in range(reproduction_Num[k]):
        #         pool.append(self.Genetics[k])
        if len(pool) == 0:
            pool.append(self.Genetics[0])
        return pool
    # 交配
    def crossover(self, Genetic1, Genetic2, sigma = 0.2):
        temp = Genetic1.w
        Genetic1.w = Genetic1.w + sigma*(Genetic1.w - Genetic2.w)
        Genetic2.w = Genetic2.w - sigma*(temp - Genetic2.w)
        return Genetic1, Genetic2
    
    # 突變
    def mutation(self, Genetic, s = -1):
        # rand = np.random.uniform(low=-1.0, high=1.0)
        # Genetic.w += s * rand
        Genetic.w *= s
        return Genetic
    # 訓練
    def fit(self, epoch):
        for i in range(epoch):
            # old_Genetics = self.Genetics.copy()
            # old_avg = self.cal_avg_fitness()

            # print("There is {} Genetics in Pool".format(len(self.Genetics)))
            # if len(set(self.Genetics)) == 1:
            #     print("Train Done")
            #     self.winner = self.Genetics[0]
            #     break

            print("epoch:{}/{}".format(i, epoch))
            if self.dump:
                print("{:-^50s}".format("Origin"))
                self.check_Genetic_w()
            # 複製
            self.Genetics = self.reproduction()
            if self.dump:
                print("{:-^50s}".format("After Reproduction"))
                self.check_Genetic_w()
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
            self.Genetics[muta_rand].w = self.mutation(self.Genetics[muta_rand]).w
            if self.dump:
                print("{:-^50s}".format("After Mutation"))
                print("Choose {}".format(muta_rand))
                self.check_Genetic_w()

            # print("loss:{}".format(self.cal_loss()))
            # print("1 / old avg :{}".format(1/old_avg))
            # print("1 / new avg :{}".format(1/self.cal_avg_fitness()))
            # if 1/old_avg < 1/self.cal_avg_fitness():
            #     self.Genetics = old_Genetics.copy()
            #     print("Choose Ex-epoch pool")
            

        # if len(set(self.Genetics)) != 1:
        fitness = np.array([self.RMSE_loss(Genetic=Genetic) for Genetic in self.Genetics])
        self.winner = self.Genetics[fitness.argmin()]
        return random.choice(self.Genetics)

    def cal_loss(self):
        loss = 1 / self.cal_avg_fitness()
        return loss
    
    def RMSE_loss(self, Genetic):
        sum = 0
        for (front, left, right, y) in self.traindata:
            predict = Genetic.predict([front, left, right])
            # print(predict)
            # predict = float(predict*80.0-40.0)
            sum += abs(y - predict)**2
        loss = (sum / len(self.traindata))**-2
        return loss



    def get_w(self, i):
        return self.Genetics[i].w
    
    def predict(self, state):
        result = self.winner.predict(state)
        print("Winner Genetic predict {}".format(result))

        return result

if __name__ == "__main__":
    ga = GeneticOpt(1, 10, dump = False)    
    print(ga.RMSE_loss(ga.Genetics[0]))
    print(ga.Genetics[0].predict([9.7355, 10.9379, 18.5740]))
    # ga.fit(epoch=100)
    # ga.predict([11.5458, 31.8026, 11.1769]) # 40
    # ga.predict([9.7355, 10.9379, 18.5740]) # -40