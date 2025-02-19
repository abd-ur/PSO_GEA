import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sigmoid function for binary conversion
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

# Modified fitness function to test multiple alpha values
def fitness_function(particle, X_train, X_test, y_train, y_test, alpha_range):
    selected_features = np.where(particle == 1)[0]

    if len(selected_features) == 0:  # Avoid empty feature set
        return 0, None

    X_train_subset = X_train[:, selected_features]
    X_test_subset = X_test[:, selected_features]

    classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)  # Using SVM for classification
    classifier.fit(X_train_subset, y_train)
    y_pred = classifier.predict(X_test_subset)

    accuracy = accuracy_score(y_test, y_pred)
    feature_ratio = len(selected_features) / X_train.shape[1]

    # Test different alpha values
    best_fitness = 0
    best_alpha = None
    for alpha in alpha_range:
        fitness = alpha * accuracy + (1 - alpha) * (1 - feature_ratio)
        if fitness > best_fitness:
            best_fitness = fitness
            best_alpha = alpha

    return best_fitness, best_alpha

# Particle Swarm Optimization for Feature Selection with Alpha Tuning
def BPSO_feature_selection(X, y, num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5):
    num_features = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the range of alpha values to test
    alpha_range = np.linspace(0.1, 0.9, 9)  # Alpha values from 0.1 to 0.9

    # Initialize particles
    particles = np.random.randint(2, size=(num_particles, num_features))  # Binary positions
    velocities = np.random.uniform(-1, 1, (num_particles, num_features))  # Velocity initialization

    pbest = particles.copy()
    pbest_fitness = np.zeros(num_particles)
    pbest_alpha = np.zeros(num_particles)

    for i in range(num_particles):
        pbest_fitness[i], pbest_alpha[i] = fitness_function(particles[i], X_train, X_test, y_train, y_test, alpha_range)

    gbest = pbest[np.argmax(pbest_fitness)]
    gbest_fitness = np.max(pbest_fitness)
    gbest_alpha = pbest_alpha[np.argmax(pbest_fitness)]

    for _ in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))

            # Apply sigmoid function for binary transformation
            probabilities = sigmoid(velocities[i])
            particles[i] = (np.random.rand(num_features) < probabilities).astype(int)

            # Evaluate fitness with the best alpha selection
            fitness, best_alpha = fitness_function(particles[i], X_train, X_test, y_train, y_test, alpha_range)

            # Update personal best
            if fitness > pbest_fitness[i]:
                pbest[i] = particles[i]
                pbest_fitness[i] = fitness
                pbest_alpha[i] = best_alpha

        # Update global best
        best_particle_idx = np.argmax(pbest_fitness)
        if pbest_fitness[best_particle_idx] > gbest_fitness:
            gbest = pbest[best_particle_idx]
            gbest_fitness = pbest_fitness[best_particle_idx]
            gbest_alpha = pbest_alpha[best_particle_idx]

    selected_features = np.where(gbest == 1)[0]
    print(f"Selected Features: {selected_features}")
    print(f"Best Fitness: {gbest_fitness}")
    print(f"Optimal Alpha: {gbest_alpha}")

    return selected_features, gbest_alpha

selected_features, best_alpha = BPSO_feature_selection(X, y)
