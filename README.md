# Tiitle : RF, XGB with GA, PSO for Fac, Kar

 - GA – (Genetic Algorithm)
 - PSO – (Particle Swarm Optimization)
 - RF – (Random Forest)
 - XGB – (XGBoost - Extreme Gradient Boosting)
 - Fac - (mfeat-fac)
 - Kar - (mfeat-kar)

---

Requirements :
 - numpy 
 - pandas  
 - scikit-learn 
 - deap 
 - pyswarms
install it by this command (pip required) :
 - pip install numpy pandas scikit-learn deap pyswarms

---

All Pseudo Code :-

1. Pseudo Code for ML (Random Forest + XGBoost)
```
LOAD dataset
SPLIT dataset into train and test sets

TRAIN Random Forest model on training data
PREDICT labels on test data using RF model
CALCULATE accuracy, precision, recall, f1_score for RF predictions

TRAIN XGBoost model on training data
PREDICT labels on test data using XGB model
CALCULATE accuracy, precision, recall, f1_score for XGB predictions
```
2. Pseudo Code for Genetic Algorithm (GA) for feature selection
```
INITIALIZE population of individuals (each individual = binary mask of features)

DEFINE fitness function: 
FOR EACH INDIVIDUAL IN POPULATION: 
SELECT features where mask bit = 1 
TRAIN Random Forest on selected features 
PREDICT on test set 
RETURN accuracy as fitness score

FOR generation in range(num_generations): 
EVALUATE fitness of all individuals 
SELECT individuals for mating (e.g., tournament selection) 
APPLY crossover to generate offspring 
APPLY mutation on offspring 
CREATE new population from offspring and/or parents

RETURN individual with best fitness (best feature subset)
TRAIN final Random Forest model on selected features
```
3. Pseudo Code for Particle Swarm Optimization (PSO) for feature selection
```
INITIALIZE swarm with n_particles, each particle is a binary vector for features

DEFINE fitness function: 
FOR each particle: 
SELECT features where particle bit > 0.5 
IF no feature selected: 
RETURN worst score 
TRAIN Random Forest on selected features 
PREDICT on test set 
RETURN (1 - accuracy) as fitness (to minimize)

FOR iteration in range(max_iterations): 
UPDATE velocity and position of each particle based on PSO rules 
EVALUATE fitness of each particle 
UPDATE personal best and global best positions

RETURN global best particle (best feature subset)
TRAIN final Random Forest model on selected features
```

---

All Results :

```
[LOADER] Loading dataset: fou . . .
[fou] Baseline evaluation with all features (76)
RF: Acc=0.8375, Prec=0.8346, Rec=0.8306, F1=0.8318
XGB: Acc=0.8325, Prec=0.8251, Rec=0.8265, F1=0.8251
============================================================
[GA] Running Genetic Algorithm . . .
[GA] Selected 37 features.

[fou] GA evaluation with Selected 37 features
RF: Acc=0.8375, Prec=0.8346, Rec=0.8306, F1=0.8318
XGB: Acc=0.8325, Prec=0.8251, Rec=0.8265, F1=0.8251
============================================================
[PSO] Running Particle Swarm Optimization . . .
[PSO] Selected 37 features.

[fou] PSO evaluation with Selected 37 features
RF: Acc=0.8375, Prec=0.8346, Rec=0.8306, F1=0.8318
XGB: Acc=0.8325, Prec=0.8251, Rec=0.8265, F1=0.8251
============================================================
[LOADER] Loading dataset: kar . . .
[kar] Baseline evaluation with all features (216)
RF: Acc=0.9650, Prec=0.9658, Rec=0.9645, F1=0.9642
XGB: Acc=0.9575, Prec=0.9558, Rec=0.9558, F1=0.9555
============================================================
[GA] Running Genetic Algorithm . . .
[GA] Selected 104 features.

[kar] GA evaluation with Selected 104 features
RF: Acc=0.9650, Prec=0.9658, Rec=0.9645, F1=0.9642
XGB: Acc=0.9575, Prec=0.9558, Rec=0.9558, F1=0.9555
============================================================
[PSO] Running Particle Swarm Optimization . . .
[PSO] Selected 132 features.

[kar] PSO evaluation with Selected 132 features
RF: Acc=0.9650, Prec=0.9658, Rec=0.9645, F1=0.9642
XGB: Acc=0.9575, Prec=0.9558, Rec=0.9558, F1=0.9555
============================================================

[SUMMARY]

--- Dataset: fou ---
RF_base: Acc=0.8375, Prec=0.8346, Rec=0.8306, F1=0.8318
XGB_base: Acc=0.8325, Prec=0.8251, Rec=0.8265, F1=0.8251
RF_ga: Acc=0.8275, Prec=0.8226, Rec=0.8192, F1=0.8197
XGB_ga: Acc=0.8000, Prec=0.7978, Rec=0.7924, F1=0.7935
RF_pso: Acc=0.8500, Prec=0.8458, Rec=0.8420, F1=0.8430
XGB_pso: Acc=0.8500, Prec=0.8487, Rec=0.8465, F1=0.8469

--- Dataset: kar ---
RF_base: Acc=0.9650, Prec=0.9658, Rec=0.9645, F1=0.9642
XGB_base: Acc=0.9575, Prec=0.9558, Rec=0.9558, F1=0.9555
RF_ga: Acc=0.9650, Prec=0.9670, Rec=0.9642, F1=0.9644
XGB_ga: Acc=0.9600, Prec=0.9594, Rec=0.9581, F1=0.9580
RF_pso: Acc=0.9675, Prec=0.9690, Rec=0.9662, F1=0.9669
XGB_pso: Acc=0.9575, Prec=0.9580, Rec=0.9569, F1=0.9564
```

![Model Performance Metrics on fou Dataset](https://github.com/user-attachments/assets/b22a66f1-fdb4-491f-97a4-9c9531c25ca7)

![Model Performance Metrics on kar Dataset ](https://github.com/user-attachments/assets/e83ab982-50cf-4009-b72e-b417cf384ae6)







