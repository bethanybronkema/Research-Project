import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from sklearn.model_selection import train_test_split
from tools import forest_classifying

'''
Initial Project Idea: Test Random Forest performace on UCI mushroom dataset
'''

def mushroom_data_format():
    # get dataset 
    mushroom = pd.read_table('agaricus-lepiota.data', sep = ',', header = None)

    #define targets and features
    targets = mushroom[0]
    features = mushroom.drop(0, axis=1)

    # define 1 as edible and 0 as poisonous
    targets_dict = {k: i for i, k in enumerate(targets.unique())}
    targets = targets.map(targets_dict)

    #update other columns to numeric values
    '''
    KEY FOR NUMERIC ASSIGNMENTS:
        Targets: poisonous - 0, edible - 1
        Cap Shape: convex - 0, bell - 1, sunken - 2, flat - 3, knobbed - 4, conical - 5
        Cap Surface: smooth - 0, scaly - 1, fibrous - 2, grooves - 3
        Cap Color: brown - 0, yellow - 1, white - 2, gray - 3, red - 4, pink - 5, buff - 6, purple - 7, cinnamon - 8, green - 9
        Bruises: true - 0, false - 1
        Odor: pungent - 0, almond - 1, anise - 2, none - 3, foul - 4, creosote - 5, fishy - 6, spicy - 7, musty - 8, 
        Gill Attachment: free - 0, attached - 1
        Gill Spacing: close - 0, crowded - 1
        Gill Size: narrow - 0, broad - 1
        Gill Color: black - 0, brown - 1, gray - 2, pink - 3, white - 4, chocolate - 5, purple - 6, red - 7, buff - 8, green - 9, yellow - 10, orange - 11
        Stalk Shape: enlarging - 0, tapering - 1
        Stalk Root: equal - 0, club - 1, bulbous - 2, rooted - 3, missing - 4
        Stalk Surface Above Ring: smooth - 0, fibrous - 1, silky - 2, scaly - 3
        Stalk Surface Below Ring: smooth - 0, fibrous - 1, scaly - 2, silky - 3
        Stalk Color Above Ring: white - 0, gray - 1, pink - 2, brown - 3, buff - 4, red - 5, orange - 6, cinnamon - 7, yellow - 8
        Stalk Color Below Ring: white - 0, pink - 1, gray - 2, buff - 3, brown - 4, red - 5, yellow - 6, orange - 7, cinnamon- 8
        Veil Type: partial - 0
        Veil Color: white - 0, brown - 1, orange - 2, yellow - 3
        Ring Number: one - 0, two - 1, none - 2
        Ring Type: pendant - 0, evanescent - 1, large - 2, flaring - 3, none - 4
        Spore Print Color: black - 0, brown - 1, purple - 2, chocolate - 3, white - 4, green - 5, orange - 6, yellow - 7, buff - 8
        Population: scattered - 0, numerous - 1, abundant - 2, several - 3, solitary - 4, clustered - 5
        Habitat: urban - 0, grasses - 1, meadows - 2, woods - 3, paths - 4, waste - 5, leaves - 6
    '''

    for j in range(1, 23):
        features_dict = {k: i for i, k in enumerate(features[j].unique())}
        features.loc[:, j] = features.loc[:, j].map(features_dict)

    #add column names to features
    features.columns = ['Cap Shape', 'Cap Surface', 'Cap Color', 'Bruises', 'Odor', 'Gill Attachment', 'Gill Spacing', 'Gill Size', 'Gill Color', 'Stalk Shape', 'Stalk Root', 'Stalk Surface Above Ring', 'Stalk Surface Below Ring', 'Stalk Color Above Ring', 'Stalk Color Below Ring', 'Veil Type', 'Veil Color', 'Ring Number', 'Ring Type', 'Spore Print Color', 'Population', 'Habitat']

    # remove Stalk Root column since it has missing values
    features.drop(['Stalk Root'], inplace = True, axis = 1)

    # save formatted data in a new file
    features.to_csv('formatted_data.csv', index=False)

    # split 2/3 training (based on reference paper)
    f_train, f_test, t_train, t_test = train_test_split(features, targets, random_state = 1, stratify = targets, train_size = 0.8)
    return f_train, f_test, t_train, t_test

#f_train, f_test, t_train, t_test = mushroom_data_format()
#forest_classifying(f_train, f_test, t_train, t_test)

'''
Updated Project Section: Compare Random Forest performances to Bayes Error
'''

from tools import gen_data
from tools import scatter_3d_data
from tools import chernoff
from tools import plot_chernoff_error

# Generate data from two different gaussian distributuions

data0, mean0, sigma0 = gen_data(500, 3, np.array([0.5, 0.5, 0.5]), 0.5)
data1, mean1, sigma1 = gen_data(500, 3, np.array([-0.5, 0.5, 0.5]), 0.5)

# Plot the data

scatter_3d_data(data0, data1)

# Plot the Chernoff Error to obtain an estimate of s

plot_chernoff_error(5000, mean0, mean1, sigma0, sigma1)

# From the plot, a value of 0.4 seems to be the lowest error
# Calculate Bhattacharyya and Chernoff error with s = 0.5

bhat_error = chernoff(0.5, mean0, mean1, sigma0, sigma1)
cher_error = chernoff(0.5, mean0, mean1, sigma0, sigma1)
#print('The Bhattacharyya error is:\n', bhat_error)
print('The Chernoff error is:\n', cher_error)

# Define features and target values

data_targets = np.zeros(data0.shape[0]+data1.shape[0])
data_features = np.zeros([data_targets.shape[0], data0.shape[1]])

for n in range(data_targets.shape[0]):
    if n < 500:
        data_targets[n] = 0
        data_features[n] = data0[n]
    else:
        data_targets[n] = 1
        data_features[n] = data1[n-500]

# Use 75 percent of the data for training and 25 percent for testing

f_train, f_test, t_train, t_test = train_test_split(data_features, data_targets, random_state = 1, stratify = data_targets, train_size = 0.75)

# Run the classifier with various estimator parameters to determine best value
num_trees = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracy = np.zeros(11)
precision = np.zeros(11) 
recall = np.zeros(11)
for trees in range(11):
        accuracy[trees], precision[trees], recall[trees] = forest_classifying(f_train, f_test, t_train, t_test, num_trees[trees])

# Plot the results

plt.subplot(3, 1, 1)
plt.scatter(num_trees, accuracy)
plt.title('Accuracy')
plt.subplot(3, 1, 2)
plt.scatter(num_trees, precision)
plt.title('Precision')
plt.subplot(3, 1, 3)
plt.scatter(num_trees, recall)
plt.title('Recall')
plt.tight_layout()
plt.show()
'''
# 30 trees seems to be the best fit for this data

for i in range(11):
    com_error = (1/num_trees[i])*cher_error
    print('The maximum accuracy for this ensemble with', num_trees[i], 'trees is:\n', "%.3f" % (1-com_error))
'''

# Numerical estimation of Bayes Error
def estimate_bayes_error():
    p = np.arange(-0.5, 0.5, step=0.012)
    q = np.arange(0, 0.5, step=0.006)
    r = np.arange(0, 0.5, step=0.006)
    xx, yy, zz = np.meshgrid(p, q, r)
    grid_samples = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    errors = []
    for i in range(grid_samples.shape[0]):
        coorx = grid_samples[i][0]
        coory = grid_samples[i][1]
        coorz = grid_samples[i][2]

        x = grid_samples[i]

        if (coorx >= 0 and coory >= 0 and coorz >= 0):
            true_class = 0
        else:
            true_class = 1

        prob_0 = scipy.stats.multivariate_normal(mean0, sigma0).pdf(x)
        prob_1 = scipy.stats.multivariate_normal(mean1, sigma1).pdf(x)

        all_prob = [prob_0, prob_1]

        if np.argmax(all_prob) == 0:
            predicted_class = 0
            errors.append(prob_1/(np.sum(all_prob)))
        else:
            predicted_class = 1
            errors.append(prob_0/(np.sum(all_prob)))
        
        if predicted_class != true_class:
            print('Mismatch!')
            print('Predicted: ', predicted_class, 'True: ', true_class)
    
    error = np.sum(errors) / len(errors)
    print('Overall Bayes Probability of Error: ', error)

# estimate_bayes_error()
# Bayes Error was found to be 0.379

def own_bayes_error():
    for i in range(data_features.shape[0]):
        errors = []
        prob0 = scipy.stats.multivariate_normal(mean0, sigma0).pdf(data_features[i])
        prob1 = scipy.stats.multivariate_normal(mean1, sigma1).pdf(data_features[i])
        all_prob = [prob0, prob1]
        if np.argmax(all_prob) == 0:
            predicted_class = 0
            errors.append(prob1/(np.sum(all_prob)))
        else:
            predicted_class = 1
            errors.append(prob0/(np.sum(all_prob)))

        if predicted_class != data_targets[i]:
            print('Mismatch!')
            print('Predicted: ', predicted_class, 'True: ', data_targets[i])
    error = np.sum(errors) / len(errors)
    print('Overall Bayes Probability of Error: ', error)

own_bayes_error()
