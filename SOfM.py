
import numpy as np
import sys, math, time
import matplotlib.pyplot as plt
from PIL import Image


# In[6]:


# Generating and normalizing data

data = (np.random.randint(0, 255, (1600,3)))
data = data / data.max()


# In[7]:


# Plotting the initial data

plt.axis('off')
plt.title("Initial State")
plt.imshow(np.reshape(data, (40,40,3)))
plt.show()


# In[8]:


# Definning the SOM class

class SOM:
    def __init__(self, alpha, epochs):

        # Initializing initial values
        self.map = np.random.uniform(0, 1, size=(40, 40, 3))
        self.initial_alpha = alpha
        self.initial_radius = 20
        self.epochs = epochs + 1
        self.landa = 0
        self.update_landa()
        self.radius = 0
        self.update_radius()
        self.alpha = 0
        self.update_alpha()
        self.influence = 0

        self.states = {}

    # Updating landa based on epoch
    def update_landa(self):
        self.landa = self.epochs / np.log(self.initial_radius)
        return

    # Updating radius based on epoch
    def update_radius(self, epoch = 0):
        self.radius = self.initial_radius * np.exp(-epoch / self.landa)
        return

    # Updating alpha based on epoch -> learning rate decay
    def update_alpha(self, epoch = 0):
        self.alpha = self.initial_alpha * np.exp(-epoch / self.landa)
        return

    def cal_distance(self, X, Y):
        return np.sum(np.square(X - Y), axis = -1, keepdims = True)

    def cal_node_distance(self, x1, x2, y1, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # Calculating neighberhood influence using Gaussians
    def cal_neighborhood_influence(self, distance):
        return np.exp(-distance / (2 * (self.radius ** 2)))

    def best_matching_unit(self, X):

        # Calculating distances of input vector X from all of SOM
        dist_matrix = self.cal_distance(X, self.map)
        dist_matrix = dist_matrix.reshape((dist_matrix.shape[0], dist_matrix.shape[1]))

        # Getting indices related to the minimum distance in the dist_matrix
        indices = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
        return indices

    # Updating weights for each SOM node if the node is within the neighborhood (radius)
    def update_map_weights(self, x_index, y_index, vector):
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                node_dist = self.cal_node_distance(i, x_index, j, y_index)
                if node_dist < self.radius:
                    influence = self.cal_neighborhood_influence(node_dist)
                    self.map[i][j] += influence * self.alpha * (vector - self.map[i][j])
        return


    # Fitting the data to the model
    def train(self, data):

        for epoch in range(self.epochs):

            # update alpha and neighborhood radius at the start of iteration
            self.update_radius(epoch)
            self.update_alpha(epoch)

            # Select a input vector
            selected_vector = data[epoch % 1600]

            # Find the winning node for the selected input vector
            min_x, min_y = self.best_matching_unit(selected_vector)

            # Updating the weights of SOM
            self.update_map_weights(min_x, min_y, selected_vector)

            if epoch % 100 == 0:
                print("Epoch no. : ", epoch)
            if epoch % 400 == 0:
                self.states[epoch] = self.map.copy()
            if epoch == self.epochs - 1:
                self.states[epoch] = self.map.copy()
        return


# In[9]:


# Declaration of SOM class and calling the train method for fitting the data

som = SOM(0.04, 10000
         )
som.train(data)


# In[ ]:


# Extract maps from SOM states dictionary for every 400 iterations and then plotting each

for epoch, net in som.states.items():
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title("Epoch no. " + str(epoch))
    plt.imshow((net * 255).astype(np.uint8))
plt.show()


# In[ ]:
