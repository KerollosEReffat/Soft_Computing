# Soft_Computing
RF, XGB with GA, PSO for mfeat-(fac, fou)

requirements :
 - numpy 
 - pandas  
 - scikit-learn 
 - deap 
 - pyswarms
install it by this command (pip required) :
 - pip install numpy pandas scikit-learn deap pyswarms

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







