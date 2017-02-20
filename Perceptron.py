import random
import numpy as np
class Perceptron:
    def __init__(self,train_data,train_num,step_size=1):
        self.train_num = train_num
        self.step_size = step_size
        self.traning_data = train_data
        weght_len = len(train_data)-1
        self.weght =[0]*weght_len
        self.bias =0
    def sign(self, x):
        if(x >= 0):
            return 1
        else:
            return -1
    def update(self,feature,lable):
        self.bias += self.step_size*lable
        for i in range(len(feature)):
            self.weght[i] += self.step_size*feature[i]*lable
        print(self.bias)
    def training(self):
        for i in range(self.train_num):
            train = random.choice(self.traning_data)
            feature = train[:-1]
            lable = train[-1]
            predict = self.sign(np.dot(self.weght,feature) +self.bias)
            print('predict:'+ str(predict))
            if lable * predict <=0:
                self.update(feature,lable)
    def predict(self,feature):
        return self.sign(np.dot(self.weght,feature) +self.bias)
    def print_parameter(self):
        for i in self.weght:
            print(str(i)+"\t")
        print('bias:'+str(self.bias))
if __name__ == '__main__':
    train_data =[[3,3,1],[4,3,1],[1,1,-1]]
    per = perceptron(train_data,100)
    per.training()
    per.print_parameter()
