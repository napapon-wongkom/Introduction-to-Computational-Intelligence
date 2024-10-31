import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = self.initialize_parameters()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def initialize_parameters(self):
        # Create an object array with the shape (number of layers - 1, 2)
        parameters = np.empty((len(self.layer_sizes) - 1, 2), dtype=object)
        
        for i in range(len(self.layer_sizes) - 1):
            # Initialize weights and biases for each layer
            W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
            b = np.zeros((1, self.layer_sizes[i + 1]))  # Biases
            parameters[i, 0] = W  # Place weights in first column
            parameters[i, 1] = b  # Place biases in second column
        return parameters

    def forward_propagation(self, X):
        A = X
        caches = []
        for i in range(len(self.parameters) - 1):
            W, b = self.parameters[i]
            Z = np.dot(A, W) + b
            A = self.sigmoid(Z)
            caches.append(A)
        W, b = self.parameters[-1]
        Z = np.dot(A, W) + b
        A = self.sigmoid(Z)
        caches.append(A)
        return caches[-1][0][0]

class PSO:
    def __init__(self,mlp,input,Y,particle,c_cognitive,c_social,inertia,max_iteration,ndays,fold):
        self.mlp = mlp
        self.input = input
        self.Y = Y
        self.particle = particle
        self.c_cognitive = c_cognitive
        self.c_social = c_social
        self.inertia = inertia
        self.max_iteration = max_iteration
        self.ndays = ndays
        self.fold = fold
        self.value_check()
        self.history = []
        self.cache_iter = []
    
    def value_check(self):
        if((self.c_cognitive + self.c_social) > 4):
            raise ValueError('sum of c_cognitive & c_social should not exceed 4')

    def objective_func(self,w):
        self.mlp.parameters = w
        Y_output = []
        for x in self.input:
            Y_output.append(self.mlp.forward_propagation(x))
        return MAE(normalize(self.Y,self.Y),Y_output)

    def rho(self,c):
        r = np.random.rand()
        return r*c

    def v_optimize(self,v,w_pbest,w_gbest,w):
        v_new = (self.inertia * v) + (self.rho(self.c_cognitive) * (w_pbest - w)) + (self.rho(self.c_social) * (w_gbest - w))
        return v_new

    def train(self):
        v = {}
        w = {}
        pbest = np.zeros(self.particle)
        for k in range(self.particle):
            v[k] = np.random.uniform(-0.1,0.1)
            w[k] = self.mlp.initialize_parameters()
            pbest[k] = self.objective_func(w[k])
        w_pbest = w
        gbest = pbest[np.argmin(pbest)]
        w_gbest = w_pbest[np.argmin(pbest)]
        iteration = 0

        while iteration <= self.max_iteration:
            for p in range(self.particle):
                score = self.objective_func(w[p])
                if score < pbest[p]:
                    pbest[p] = score
                    w_pbest[p] = w[p]
                if score < gbest:
                    gbest = score
                    w_gbest = w[p]
                v[p] = self.v_optimize(v[p],w_pbest[p],w_gbest,w[p])
                w[p] = w[p] + v[p]
            if (iteration > 0):
                print(f"iteration : {iteration}/{self.max_iteration} gobal best : {gbest}")
            self.cache_iter.append(iteration)
            self.history.append(gbest)
            iteration += 1
        self.mlp.parameters = w_gbest
    
    def test(self, Xtest, Ytest, buffer):
        predicts = []
        for x in Xtest:
            predict = self.mlp.forward_propagation(x)
            predicts.append(predict)
        predicts = np.array(predicts)
        nor_Ytest = normalize(Ytest,Ytest)
        mean_abs_error = MAE(nor_Ytest, predicts)
        buffer.append(mean_abs_error)
        plt.figure()
        plt.title(f"Compare Predict & Desired Value for {self.ndays} days Fold:{self.fold+1}")
        plt.plot(nor_Ytest, label="Actual")
        plt.plot(predicts, label="Predicted")
        plt.ylabel("Benzene concentration")
        plt.xlabel("Samples")
        plt.legend()
        plt.figtext(0.75, 0.06, f"Mean Absolute Error: {mean_abs_error:.4f}", ha="center", fontsize=10)
        print(f"MAE of output is : {mean_abs_error}")

    def plot(self):
        plt.figure()
        plt.title(f'Simulation Predict {self.ndays} days  Fold:{self.fold+1}')
        plt.plot(self.cache_iter,self.history)
        plt.xlabel('iteration')
        plt.ylabel('gbest')
        plt.grid()

    def bar_plot(self,MAE_set):
        x_set = []
        for i in range(10):
            x_set.append(f"Fold {i+1}")
        average_MAE = np.mean(MAE_set)
        x = x_set
        y = MAE_set
        plt.figure()
        plt.title(f'MAE Average Graph {self.ndays} days')
        plt.bar(x,y,color='skyblue', label="MAE")
        plt.axhline(y=average_MAE, color='red', linestyle='--', label=f"Average MAE ({average_MAE:.4f})")
        plt.xlabel("Fold")
        plt.ylabel("MAE in each fold")
        plt.legend()
        plt.grid()

def import_data(file):
    """
    function to import and tranfer data to array
    """
    data = pd.read_excel(file)
    array_data = data.to_numpy()
    new_data = []
    attribute = [3,6,8,10,11,12,13,14,5]
    for i in range(len(data)):
        data_buffer = []
        for j in range(len(attribute)):
            data_buffer.append(array_data[i][attribute[j]])
        new_data.append(data_buffer)
    data_set = np.array(new_data)
    return data_set

def k_fold_validation(data_set, i , k):
    np.random.shuffle(data_set)
    X = data_set[ : , : -1]
    Y = data_set[ : ,-1]
    Xset = np.array_split(X,k)
    Yset = np.array_split(Y,k)
    Xtest = Xset[i]
    Ytest = Yset[i]
    Xtrain = np.concatenate([Xset[j] for j in range(k) if j != i])
    Ytrain = np.concatenate([Yset[j] for j in range(k) if j != i])
    return Xtrain, Ytrain, Xtest, Ytest

def MAE(true_Y,Y):
    true_Y = np.array(true_Y)
    Y = np.array(Y)
    MAE = np.mean(np.absolute(true_Y - Y))
    return MAE

def normalize(data,value):
    """
    function to normalize value to [0,1]
    """
    min_val = np.min(data)
    max_val = np.max(data)
    result = (value - min_val) / (max_val - min_val)
    return result

def denormalize(data,value):
    min_val = np.min(data)
    max_val = np.max(data)
    result = value * (max_val - min_val) + min_val
    return result

def create_dataset_for_prediction(data, look_ahead):
    """
    Function to create dataset for multi-day look-ahead prediction.
    """
    X = data[ : , :-1]
    Y = data[ : ,-1]
    Y = np.roll(Y,-24 * look_ahead)
    new_data = np.concatenate((X,Y.reshape(-1,1)),axis=1)
    new_data = new_data[ : -24 * look_ahead]
    return new_data
    
    
if __name__ == '__main__':
    print("_____________________________________________________________________________")
    plt.close("all")
    file = 'AirQualityUCI.xlsx'
    k = 10 # number of fold
    data_set = import_data(file)
    data_5day = create_dataset_for_prediction(data_set,5)
    data_10day = create_dataset_for_prediction(data_set,10)
    layer_size = [8,16,1]
    particle = 10
    c_cognitive = 1.5
    c_social = 1.5
    inertia = 0.7
    max_iteration = 50
    plot_buffer_5day = []
    plot_buffer_10day = []
    # #=============================================================================
    for f in range(k):
        print("======================================")
        print(f"Fold : {f+1}")
        Xtrain5day,Ytrain5day,Xtest5day,Ytest5day = k_fold_validation(data_5day,f,k)
        Xtrain10day,Ytrain10day,Xtest10day,Ytest10day = k_fold_validation(data_10day,f,k)
        mlp5day = MLP(layer_size)
        pso_5day = PSO(mlp5day,Xtrain5day,Ytrain5day,particle,c_cognitive,c_cognitive,inertia,max_iteration,5,f)
        pso_5day.train()
        pso_5day.test(Xtest5day,Ytest5day,plot_buffer_5day)
        pso_5day.plot()
        mlp10day = MLP(layer_size)
        pso_10day = PSO(mlp10day,Xtrain10day,Ytrain10day,particle,c_cognitive,c_cognitive,inertia,max_iteration,10,f)
        pso_10day.train()
        pso_10day.test(Xtest10day,Ytest10day,plot_buffer_10day)
        pso_10day.plot()
    pso_5day.bar_plot(plot_buffer_5day)
    pso_10day.bar_plot(plot_buffer_10day)
    plt.show()
        