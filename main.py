import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import parallel_backend
from deap import base, creator, tools, algorithms
import pyswarms as ps
import random
import matplotlib.pyplot as plt

# ===== CONFIGURATION =====
DATASETS = ['fou', 'kar']  # 'fou' أو 'kar'
USE_RF = True
USE_XGB = True
USE_GA = True
USE_PSO = True

# ===== DATASET URLS =====
URLS = {
    'fou': "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-fou",
    'kar': "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-fac",
}

# ===== DATA LOADER =====
def load_data(name):
    print(f"[LOADER] Loading dataset: {name} . . .")
    X = pd.read_csv(URLS[name], sep=r"\s+", header=None).values
    y = np.repeat(np.arange(10), X.shape[0] // 10)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ===== BASE RF TRAINING =====
def train_rf_es(X_tr, y_tr, X_te, y_te, start=50, step=50, max_est=300, patience=3, tol=1e-3):
    clf = RandomForestClassifier(n_estimators=start, warm_start=True, random_state=42, n_jobs=-1)
    best, no_imp = 0, 0
    while clf.n_estimators <= max_est:
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        # print(f"[RF] trees={clf.n_estimators}, acc={acc:.4f}")
        if acc > best + tol:
            best, no_imp = acc, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
        clf.n_estimators += step
    return clf

# ===== BASE XGB TRAINING =====
def train_xgb(X_tr, y_tr, X_te, y_te, n_est=300, es_rounds=3):
    model = XGBClassifier(n_estimators=n_est, use_label_encoder=False, eval_metric='mlogloss',
                          early_stopping_rounds=es_rounds, verbosity=0, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    # print(f"[XGB] Best iteration: {model.best_iteration}")
    return model

# ===== GENETIC ALGORITHM FEATURE SELECTION =====
def train_ga(X_train, y_train, X_test, y_test):
    print("[GA] Running Genetic Algorithm . . .")
    n_features = X_train.shape[1]

    # Delete if already exists to avoid warning
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(ind):
        idx = [i for i, bit in enumerate(ind) if bit]
        if not idx:
            return 0.0,
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        model.fit(X_train[:, idx], y_train)
        pred = model.predict(X_test[:, idx])
        return accuracy_score(y_test, pred),

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
                        stats=stats, halloffame=hof, verbose=False)

    best_ind = hof[0]
    selected = [i for i, bit in enumerate(best_ind) if bit]
    print(f"[GA] Selected {len(selected)} features.")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X_train[:, selected], y_train)
    model.selected_features = selected
    return model

# ===== PSO FEATURE SELECTION =====
def train_pso(X_train, y_train, X_test, y_test):
    print("[PSO] Running Particle Swarm Optimization . . .")
    n_particles = 30
    dimensions = X_train.shape[1]

    def fitness_func(swarm):
        scores = []
        for particle in swarm:
            mask = particle > 0.5
            if not mask.any():
                scores.append(1.0)
                continue
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            model.fit(X_train[:, mask], y_train)
            pred = model.predict(X_test[:, mask])
            acc = accuracy_score(y_test, pred)
            scores.append(1 - acc)
        return np.array(scores)

    # options = {'c1': 2, 'c2': 2, 'w': 0.9}
    options={'c1':1.5, 'c2':1.5, 'w':0.7, 'k':3, 'p':2}
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)
    best_cost, best_pos = optimizer.optimize(fitness_func, iters=10, verbose=False)

    selected = np.where(best_pos > 0.5)[0]
    print(f"[PSO] Selected {len(selected)} features.")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X_train[:, selected], y_train)
    model.selected_features = selected
    return model

# ===== EVALUATION =====
def evaluate_with_feature_selection(name, X_tr, X_te, y_tr, y_te):
    print(f"[{name}] Baseline evaluation with all features ({X_tr.shape[1]})")
    # Baseline with all features
    rf_base = train_rf_es(X_tr, y_tr, X_te, y_te)
    xgb_base = train_xgb(X_tr, y_tr, X_te, y_te)
    
    # Evaluate baseline
    results = {}
    for label, model in (('RF', rf_base), ('XGB', xgb_base)):
        y_pred = model.predict(X_te)
        results[f"{label}_base"] = {
            'accuracy': accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_te, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_te, y_pred, average='macro', zero_division=0),
        }
        print(f"{label}: Acc={results[f'{label}_base']['accuracy']:.4f}, Prec={results[f'{label}_base']['precision']:.4f}, "
                  f"Rec={results[f'{label}_base']['recall']:.4f}, F1={results[f'{label}_base']['f1']:.4f}")

    print("============================================================")
    
    # GA feature selection
    ga_model = train_ga(X_tr, y_tr, X_te, y_te)  # ترجع نموذج مدرب
    selected_ga = ga_model.selected_features    # قائمة الميزات المختارة
    print(f"\n[{name}] GA evaluation with Selected {len(selected_ga)} features")
    X_tr_ga = X_tr[:, selected_ga]
    X_te_ga = X_te[:, selected_ga]
    rf_ga = train_rf_es(X_tr_ga, y_tr, X_te_ga, y_te)
    xgb_ga = train_xgb(X_tr_ga, y_tr, X_te_ga, y_te)
    
    for label, model in (('RF', rf_ga), ('XGB', xgb_ga)):
        y_pred = model.predict(X_te_ga)
        results[f"{label}_ga"] = {
            'accuracy': accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_te, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_te, y_pred, average='macro', zero_division=0),
        }
        print(f"{label}: Acc={results[f'{label}_base']['accuracy']:.4f}, Prec={results[f'{label}_base']['precision']:.4f}, "
                  f"Rec={results[f'{label}_base']['recall']:.4f}, F1={results[f'{label}_base']['f1']:.4f}")
        
    print("============================================================")

    # PSO feature selection
    pso_model = train_pso(X_tr, y_tr, X_te, y_te)
    selected_pso = pso_model.selected_features
    print(f"\n[{name}] PSO evaluation with Selected {len(selected_pso)} features")
    X_tr_pso = X_tr[:, selected_pso]
    X_te_pso = X_te[:, selected_pso]
    rf_pso = train_rf_es(X_tr_pso, y_tr, X_te_pso, y_te)
    xgb_pso = train_xgb(X_tr_pso, y_tr, X_te_pso, y_te)
    
    for label, model in (('RF', rf_pso), ('XGB', xgb_pso)):
        y_pred = model.predict(X_te_pso)
        results[f"{label}_pso"] = {
            'accuracy': accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_te, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_te, y_pred, average='macro', zero_division=0),
        }
        print(f"{label}: Acc={results[f'{label}_base']['accuracy']:.4f}, Prec={results[f'{label}_base']['precision']:.4f}, "
                  f"Rec={results[f'{label}_base']['recall']:.4f}, F1={results[f'{label}_base']['f1']:.4f}")

    return results

def plot_results_bar_chart(all_results):
    for dataset, models in all_results.items():
        methods = list(models.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        # تجهيز بيانات القياسات لكل مقياس ولكل طريقة
        data = []
        for metric in metrics:
            data.append([models[m][metric] for m in methods])

        x = np.arange(len(methods))  # مواقع الأعمدة على المحور X
        width = 0.2  # عرض كل عمود

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, data[i], width, label=metric.capitalize())

        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title(f'Model Performance Metrics on {dataset}')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(methods)
        ax.set_ylim([0, 1])  # القيم بين 0 و 1
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()

# ===== MAIN =====
def main():
    all_results = {}
    for ds in DATASETS:
        X_tr, X_te, y_tr, y_te = load_data(ds)
        all_results[ds] = evaluate_with_feature_selection(ds, X_tr, X_te, y_tr, y_te)
        print("============================================================")
    
    print("\n[SUMMARY]")
    for ds, models in all_results.items():
        print(f"\n--- Dataset: {ds} ---")
        for m, res in models.items():
            print(f"{m}: Acc={res['accuracy']:.4f}, Prec={res['precision']:.4f}, "
                  f"Rec={res['recall']:.4f}, F1={res['f1']:.4f}")

    plot_results_bar_chart(all_results)
    
    return all_results

if __name__ == "__main__":
    results = main()
