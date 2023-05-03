import numpy as np
import RBFN
import random
class GeneticOpt(object):
    def __init__(self, G_num = 5, RBFN_K = 10, dump = False) -> None:
        self.Genetics = []
        self.pool = []
        self.traindata = [dist for dist in RBFN.getTrain4d()]
        self.RBFN = RBFN.RBFNet(RBFN_K)
        
        self.dump = dump
        self.is_done = False
        self.Muta_Fra = 0.3
        self.Cross_Fra = 0.3
        self.REPET_fra= 0.2
        self.POOL_NUM = 15

        for i in range(G_num):
            weight = np.random.randn(RBFN_K + 1) * np.random.randn(RBFN_K + 1)
            self.Genetics.append(weight)
        print(self)
    # def cal_fitness(self, Genetic):
    #     sum = 0
    #     for (front, left, right, y) in self.traindata:
    #         predict = Genetic.predict([front, left, right])
    #         # print(predict)
    #         # predict = float(predict*80.0-40.0)
    #         sum += abs(y - predict)
    #         # sum += (y - predict)
    #     return 1/sum
    
    # def cal_fitness_v2(self):
    #     apha = 0.2
    #     self.objective_values = [self.cal_fitness(Genetic) for Genetic in self.Genetics]
    #     min_obj_value = min(self.objective_values)
    #     max_obj_value = max(self.objective_values)
    #     fitness = []
    #     for value in self.objective_values:
    #         fitness.append(max(apha*(max_obj_value-min_obj_value), 10**-5) + (max_obj_value - value))
    #     avg_fitness = np.mean(fitness)
    #     return fitness, avg_fitness


    # def cal_avg_fitness(self):
    #     fitness = []
    #     for Genetic in self.Genetics:
    #         fitness.append(self.cal_fitness(Genetic))
    #     return np.mean(fitness)
    
    def check_Genetic_w(self):
        print("Genetic Pool :")
        for i, Genertic in enumerate(self.Genetics):
            print("Genertic {} weight:{}".format(i, Genertic))

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
        WINER_NUM = int(self.POOL_NUM * self.REPET_fra)
        # print("sorted_fitness arg before repr:{}".format(sorted_fitness))
        sorted_fitness[-WINER_NUM:] = sorted_fitness[0]
        # print("sorted_fitness arg after repr:{}".format(sorted_fitness))
        for index in sorted_fitness:
            pool.append(self.Genetics[index])
            
        # print("POOL NUM : {}".format(len(pool)))
        # print("sorted fitness arg:{}".format(sorted_fitness))

        if len(pool) == 0:
            pool.append(self.Genetics[0])
        return pool
    # 交配
    def crossover(self, Genetic1, Genetic2, sigma = 0.2):
        temp = Genetic1
        Genetic1 = Genetic1 + sigma*(Genetic1 - Genetic2)
        Genetic2 = Genetic2 - sigma*(temp - Genetic2)
        return Genetic1, Genetic2
    
    # 突變
    def mutation(self, Genetic, s = -1):
        rand = np.random.uniform(low=-1.0, high=1.0)
        Genetic += s * rand
        # Genetic *= s 
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
            for i in range(int(self.POOL_NUM * self.Cross_Fra // 2)):
                rand_selet = random.sample(rand_list, k=2)
                self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]] =  self.crossover(self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]])
                if self.dump:
                    print("{:-^50s}".format("After Crossover"))
                    print("Choose {} and {}".format(rand_selet[0], rand_selet[1]))
                    self.check_Genetic_w()
            
            # 突變
            for i in range(int(self.POOL_NUM * self.Muta_Fra)):
                muta_rand = random.choice(rand_list)
                self.Genetics[muta_rand] = self.mutation(self.Genetics[muta_rand])
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
        self.RBFN.w = self.winner
    
    def RMSE_loss(self, Genetic):
        sum = 0
        # self.RBFN.w = Genetic
        for (front, left, right, y) in self.traindata:
            predict = self.predict([front, left, right], Genetic)
            # print(predict)
            # predict = float(predict*80.0-40.0)
            sum += abs(y - predict)**2
        loss = (sum / len(self.traindata))**(1/2)
        return loss



    def get_w(self, i):
        return self.Genetics[i]
    
    def predict(self, state, weight):
        self.RBFN.w = weight
        result = self.RBFN.predict(state)

        return result

if __name__ == "__main__":
    ga = GeneticOpt(30, 10, dump = False)    
    # print(ga.RMSE_loss(ga.Genetics[0]))
    print(np.shape(ga.Genetics))
    # print(ga.Genetics[0].predict([9.7355, 10.9379, 18.5740]))
    ga.fit(epoch=100)
    winner  = ga.winner
    ga.predict([11.5458, 31.8026, 11.1769], winner) # 40
    ga.predict([9.7355, 10.9379, 18.5740], winner) # -40