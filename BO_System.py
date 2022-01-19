import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from GP_System import MyGP

seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def normalize(value):
    value = (value - torch.min(value)) / (torch.max(value) - torch.min(value))
    return value


class SurrogateModel():
    def __init__(self, bound_x, train_x, train_y, num,
                 data_dim=140, train_times=10):
        self.data_dim = data_dim
        self.num = num
        self.train_time = train_times
        self.x = bound_x
        self.train_x, self.train_y = train_x, train_y
        self.model = self.initialize_model(train_x=self.train_x, train_obj=self.train_y)
        self.train()

    def initialize_model(self, train_x, train_obj):
        model = MyGP(train_x, train_obj, self.data_dim, self.train_time)
        return model

    def train(self):
        self.model.train()

    def getDataAtX(self, x):
        test_dataset = TensorDataset(x)
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
        means = torch.tensor([0.])
        with torch.no_grad():
            for x_batch in test_loader:
                pred = self.model.test(x_batch[0])
                means = torch.cat([means, pred.mean])
        means = means[1:]
        return means

    def getPredNum(self, num):
        mean = self.getDataAtX(self.x)
        mean = mean.reshape(-1).numpy().tolist()
        index = [i for i in range(len(mean))]
        indexAndMean = zip(index, mean)
        sorted_data_index = sorted(indexAndMean, key=lambda x: x[1], reverse=True)
        result_index = zip(*sorted_data_index)
        index, mean = [list(x) for x in result_index]
        return index[0:num]

    def optimization(self):
        pred_index = self.getPredNum(self.num)
        return pred_index


def getFirst(i, x, y):
    integer = i
    y_next = y[integer:integer + 1]
    x_next = x[integer:integer + 1]
    return x_next, y_next


def getTrainData(n, x, y):
    train_x, train_y = None, None
    for i in range(n):
        x_next, y_next = getFirst(i, x, y)
        if train_x is None:
            if (y_next != y_next).item() != 1:
                train_x = x_next
                train_y = y_next
        else:
            if (y_next != y_next).item() != 1:
                train_x = torch.vstack([train_x, x_next])
                train_y = torch.vstack([train_y, y_next])
    train_y = train_y.reshape(-1)
    return train_x, train_y


def main():
    with open("data.txt", "r") as f:
        weidu = int(f.readline())
        num_data = int(f.readline())
        num_key = int(f.readline())
        num_initial = int(f.readline())
    f.close()

    df = pd.read_csv('case1.output/x.csv')
    my_array = np.array(df)
    my_tensor = torch.from_numpy(my_array).float()
    y = my_tensor[:num_data, 1:2]
    x = my_tensor[:num_data, 2:my_tensor.shape[1] - 1]
    x = normalize(x)
    y = y.float()

    train_x, train_y = getTrainData(num_initial, x, y)

    model = SurrogateModel(bound_x=x, train_x=train_x, train_y=train_y, num=num_key, data_dim=x.shape[1],
                           train_times=50)
    pred_index = model.optimization()
    print(pred_index)
    with open("result_py.txt", "w") as f:
        for i in pred_index:
            f.write(str(i) + '\n')
    f.close()


if __name__ == "__main__":
    main()
