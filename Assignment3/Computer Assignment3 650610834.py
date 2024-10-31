import numpy as np
import random
import matplotlib.pyplot as plt

# MLP class with adjustable layers and biases
class MLP:
    def __init__(self, layer_sizes):
        self.layers = layer_sizes
        self.weights = [np.random.uniform(-1, 1, (self.layers[i], self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.biases = [np.random.uniform(-1, 1, (1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    # Activation function and derivative (Sigmoid)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = self.sigmoid(np.dot(x, w) + b)
            self.activations.append(x)
        return x

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, x_input, d_output, layer_sizes, pop_size, generations, mutation_rate):
        self.x_input = x_input
        self.d_output = d_output
        self.layer_sizes = layer_sizes
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        MLP_pop = []
        for _ in range(self.pop_size):
            MLP_pop.append(MLP(self.layer_sizes))
        return MLP_pop

    def mutate(self, weights, biases):
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, 0.5, weights[i].shape)
            if random.random() < self.mutation_rate:
                biases[i] += np.random.normal(0, 0.5, biases[i].shape)
        return weights, biases

    def crossover(self, parent1, parent2):
        child_weights = []
        child_biases = []
        for w1, w2, b1, b2 in zip(parent1.weights, parent2.weights, parent1.biases, parent2.biases):
            mask = np.random.rand(*w1.shape) < 0.5
            child_weights.append(np.where(mask, w1, w2))
            mask_bias = np.random.rand(*b1.shape) < 0.5
            child_biases.append(np.where(mask_bias, b1, b2))
        return child_weights, child_biases

    def train(self):
        population = self.initialize_population()
        history = []

        for gen in range(self.generations):
            # Evaluate fitness (accuracy in this case)
            fitness_scores = []
            for mlp in population:
                predictions = np.round([mlp.forward(x) for x in self.x_input])
                fitness_scores.append(np.mean(predictions.flatten() == self.d_output))

            # Select best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices[:self.pop_size // 2]]

            # Crossover and mutation
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = random.sample(population, 2)
                child_weights, child_biases = self.crossover(parent1, parent2)
                child_weights, child_biases = self.mutate(child_weights, child_biases)
                child = MLP(self.layer_sizes)
                child.weights = child_weights
                child.biases = child_biases
                new_population.append(child)

            population.extend(new_population)
            history.append(max(fitness_scores))
            print(f"Generation {gen + 1}, Best Fitness: {max(fitness_scores)}")

        return population[0], history

# Load data function (assumes wdbc.data is in the same directory)
def load_data(file_name):
    data = np.genfromtxt(file_name, delimiter=',', dtype=str)
    ids, diagnosis, x_input = data[:, 0], data[:, 1], data[:, 2:].astype(np.float32)
    d_output = []
    for d in diagnosis:     # M=1, B=0
        if(d == 'M'):
            d_output.append(1)
        elif(d == 'B'):
            d_output.append(0)
    d_output = np.array(d_output)
    return x_input, d_output

def k_fold_validation(x_input, d_output, k=10):
    fold_size = len(x_input) // k
    indices = np.arange(len(x_input))
    np.random.shuffle(indices)
    
    folds_x = np.array_split(x_input[indices], k)
    folds_y = np.array_split(d_output[indices], k)
    
    for i in range(k):
        # Create training and testing sets for the i-th fold
        x_test = folds_x[i]
        y_test = folds_y[i]
        x_train = np.concatenate([folds_x[j] for j in range(k) if j != i])
        y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
        
        yield x_train, y_train, x_test, y_test  # Yielding for each fold

def plot_confusion_matrix(desired_output, prediction):
    # Ensure the arrays are integers and binary (0, 1)
    y_true = np.array(desired_output, dtype=int).flatten()
    y_pred = np.array(prediction, dtype=int).flatten()

    # Initialize the confusion matrix for binary classification
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        # Only update the matrix if values are within expected binary range
        if true in [0, 1] and pred in [0, 1]:
            confusion_matrix[true, pred] += 1

    TP = confusion_matrix[1, 1]  # True Positive
    FP = confusion_matrix[0, 1]  # False Positive
    FN = confusion_matrix[1, 0]  # False Negative
    TN = confusion_matrix[0, 0]  # True Negative

    # Calculate accuracy, handling the zero denominator
    total = TP + TN + FP + FN
    accuracy_rate = (100 * (TP + TN) / total)
    acc_text = "Accuracy : {:.2f} %".format(accuracy_rate)
    # Plot the confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    # Annotate the confusion matrix with counts
    for (i, j), val in np.ndenumerate(confusion_matrix):
        plt.text(j, i, val, ha='center', va='center', color='red')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.text(0.85,1.60,s = acc_text)
    plt.xticks([0, 1], ['B', 'M'])
    plt.yticks([0, 1], ['B', 'M'])
    
    return accuracy_rate

def average_plot(acc,k):
    plt.figure()
    print("Accuracy on each folds : {} %".format(acc))
    average_accuracy = np.mean(acc)
    print(f"Average Accuracy over {k} folds: {average_accuracy :.2f}%")
    x = []
    y = acc
    for i in range(k):
        x.append(f'Fold {i+1}')
    plt.bar(x,y,label = 'Accuracy')
    plt.axhline(y=average_accuracy, color='red', linestyle='--', label=f"Average Accuracy ({average_accuracy:.2f})")
    plt.xlabel("Folds")
    plt.ylabel("Accuracy on each fold")
    plt.title("Accuracy of all fold")
    plt.legend()
    plt.grid()

if __name__ == '__main__':
    # Main execution
    file = 'wdbc.txt'
    X, Y = load_data(file)
    #initialize parameters
    k = 10  # Number of folds
    layer_sizes = [30, 30, 1]  # [input_size, hidden_layers, output_size]
    population = 60
    generation = 100
    mutation_rate = 0.01
    accuracies = []
    plot = 1
    for train_X, train_Y, test_X, test_Y in k_fold_validation(X, Y, k):
        print('___________________________________')
        print('Fold : {}'.format(plot))
        print(' ')
        # Genetic Algorithm to find the best weights for each fold
        GA = GeneticAlgorithm(train_X, train_Y, layer_sizes, population, generation, mutation_rate)
        best_mlp, history = GA.train()
        # Evaluate the final model for the current fold
        test_predictions = []
        predict = []
        for x in test_X:
            test_predictions.append(np.round(best_mlp.forward(x)))
        test_predictions = np.array(test_predictions)
        accuracy = plot_confusion_matrix(test_Y,test_predictions.flatten())
        accuracies.append(accuracy)
        plt.figure(0)
        plt.subplot(1,10,plot)
        plt.grid()
        if(plot == 1):
            plt.ylabel('Best Fitness')
        plt.xlabel('Fold {}'.format(plot))
        plt.plot(history)
        plot+=1
    
    plt.suptitle('Genetic Algorithm Progress')
    average_plot(accuracies,k)
    plt.show()