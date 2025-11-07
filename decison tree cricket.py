
# Cricket Team Player Selection using Decision Tree
# Step-by-step code for beginners

# STEP 1: Import Required Libraries
# ===================================
import numpy as np  # For numerical operations and arrays
from sklearn.tree import DecisionTreeClassifier  # The decision tree algorithm
from sklearn.model_selection import train_test_split  # To split data
from sklearn import tree  # To visualize tree structure
from sklearn.metrics import accuracy_score  # To measure accuracy

# STEP 2: Create the Dataset
# ===========================
# We create performance data for 50 cricket players
# Features: [Wickets, Runs, Fielding, Fitness Level]

np.random.seed(42)  # For reproducible results
n_players = 50

# Generate random player statistics
wickets = np.random.randint(0, 25, n_players)
runs = np.random.randint(50, 550, n_players)
fielding = np.random.randint(0, 16, n_players)
fitness = np.random.randint(3, 11, n_players)

# Combine all features into one array
player_data = np.column_stack([wickets, runs, fielding, fitness])

# STEP 3: Create Selection Labels (Target Variable)
# ==================================================
# Based on complex criteria, decide if player should be selected
selection_status = []
for i in range(n_players):
    w, r, f, fit = player_data[i]

    # Selection criteria - multiple conditions
    if w >= 15 and r >= 350:
        selection_status.append(1)  # Selected
    elif w >= 12 and r >= 300 and f >= 8:
        selection_status.append(1)
    elif w >= 10 and r >= 250 and fit >= 8 and f >= 6:
        selection_status.append(1)
    elif r >= 400 and fit >= 9 and f >= 10:
        selection_status.append(1)
    else:
        selection_status.append(0)  # Not selected

selection_status = np.array(selection_status)

print("Dataset created with", n_players, "players")
print("Selected players:", np.sum(selection_status))
print("Not selected:", len(selection_status) - np.sum(selection_status))

# STEP 4: Split Data into Training and Testi
