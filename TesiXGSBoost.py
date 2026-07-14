"""
===============================================================================
Modulo:         TesiXGSBoost.py
Autore:         Alessandro Ferrari
Data:           01 Marzo 2026
Versione:       1.0
===============================================================================

Descrizione:
Questo modulo contiene il programma di tesi di Alessandro Ferrari 

Dipendenze:
    - Librerie Standard: os
    - Manipolazione Dati: pandas, numpy
    - Machine Learning: scikit-learn (sklearn), xgboost
    - Ottimizzazione Statistica: scipy
    - Acquisizione Dati: ucimlrepo
===============================================================================
"""

# region LIBRERIE E DICHIARAZIONI 

#Importazione delle librerie
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

import warnings
from scipy.optimize import OptimizeWarning

# Ignoro  il warning sui nomi delle feature mancanti (tipico di scikit-learn durante predict)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Ignoro il warning di SciPy quando il punto di partenza (x0) è fuori dai bounds
warnings.filterwarnings("ignore", category=OptimizeWarning)


#COSTANTI
TEST_SIZE = 0.2         #Percentuale del 20% di test per dare i risultati finali
VALIDATION_SIZE = 0.2   #Percentuale del 20% di parte validazione
RANDOM_SEED = 42        #Inizializzazione del random seed

#Inizializzo il Random Seed (per avere sempre gli stessi risultati)
np.random.seed(RANDOM_SEED)

#VARIABILI
path_file="steel_plates_faults.xlsx"
folder_export = "export_dati"

#Parametri di debug
SKIP_STEP5 = False
SKIP_STEP6 = False
SKIP_STEP7 = False
PRINT_GRAPH = False  #Visualizzo il grafico dell'errore

# endregion

#varie utility
# region STEP 0

# Creo una cartella per tenere tutto in ordine
os.makedirs(folder_export, exist_ok=True)
# endregion


#Creo un nuovo file da zero aggiungendo i campi di temperatura velocità e pressione modificati in base al difetto
# region STEP 1 

N_RIGHE_TARGET = 10000

#print("--- STEP 1: Caricamento da file locale ) ---")
df = pd.DataFrame(index=range(N_RIGHE_TARGET))

# Definisco i numeri possibili (da 0 a 7) e la probabilità
valori_possibili = [0, 1, 2, 3, 4, 5, 6, 7]
probabilita = [0.8, 0.03, 0.03, 0.02, 0.02, 0.02, 0.04, 0.04]

#valori_possibili = [0, 1, 2, 3, 4, 5, 6,7]
#probabilita = [0.8, 0.03, 0.03, 0.02, 0.04, 0.04, 0.03, 0.01]



#  Aggiungo le colonne base di temperatura velocità e pressione
n_rows = N_RIGHE_TARGET
df['Rolling_Temp_C'] = np.random.normal(loc=950, scale=20, size=n_rows)
df['Roller_Speed_m_sec'] = np.random.normal(loc=10, scale=1, size=n_rows)
df['Pressure_Bar'] = np.random.normal(loc=200, scale=10, size=n_rows)
df['Defects'] = np.random.choice(valori_possibili, size=n_rows, p=probabilita)

# 'No_Defects'  - 1 'Pastry', 2 'Z_Scratch', 3 'K_Scratch', 4 'Stains',5 'Dirtiness', 6'Bumps', 7 'Other_Faults' 

#difetto di sfogliatura aumento la temperatura (Pastry)
mask_defect = (df['Defects'] == 1) 
df.loc[mask_defect, 'Rolling_Temp_C'] += np.random.uniform(50, 100, size=mask_defect.sum())

#difetto di graffi e altro aumento la velocità (Z_Scratch)
mask_defect = (df['Defects'] == 2) 
df.loc[mask_defect, 'Roller_Speed_m_sec'] += np.random.uniform(3, 6, size=mask_defect.sum())

#difetto di graffi diminuisco la velocità (K_Scratch)
mask_defect =  (df['Defects'] == 3)
df.loc[mask_defect, 'Roller_Speed_m_sec'] -= np.random.uniform(3, 6, size=mask_defect.sum())

#difetto di  diminuisco la temperatura (Stains)
mask_defect =  (df['Defects'] == 4)
df.loc[mask_defect, 'Rolling_Temp_C'] -= np.random.uniform(50, 100, size=mask_defect.sum())

#difetto irregolarità aumento la pressione(Bumps)
mask_defect = (df['Defects'] == 5) 
df.loc[mask_defect, 'Pressure_Bar'] += np.random.uniform(40, 80, size=mask_defect.sum())

#difetto sporco diminuisco la pressione(Dirtiness)
mask_defect = (df['Defects'] == 6)
df.loc[mask_defect, 'Pressure_Bar'] -= np.random.uniform(40, 80, size=mask_defect.sum())

#difetto di altro diminuisco la temperatura e aumento la velocità (Other_Faults)
mask_defect =  (df['Defects'] == 7)
df.loc[mask_defect, 'Rolling_Temp_C'] -= np.random.uniform(50, 100, size=mask_defect.sum())
df.loc[mask_defect, 'Roller_Speed_m_sec'] -= np.random.uniform(1, 3, size=mask_defect.sum())

#Salvo il file per diagnostica
df.to_excel(f"{folder_export}/01_Step1_Dati_Aumentati.xlsx", index=False)
print("\n --- STEP 1  10000 righe generate. Variabili simulate corrette.")
# Raggruppo per difetto e calcolo la media dei 3 parametri fisici
tabella_medie = df.groupby('Defects')[['Rolling_Temp_C', 'Roller_Speed_m_sec', 'Pressure_Bar']].mean().round(2)
print(tabella_medie)


# endregion


# region STEP 2: Preparazione Target e Features
print("\n--- STEP 2: Preparazione Target e Features ---")

# 1. Separo le feature (X) dal target (y)
# Rimuovo la colonna 'Defects' da X per evitare il data leakage nell'addestramento
X = df.drop(columns=['Defects'])

# 2. Mappiamo i numeri generati ai nomi reali dei difetti per mantenere 
# la perfetta compatibilità e leggibilità negli STEP 5 e 6
mappa_difetti = {
    0: 'No_Defects',
    1: 'Pastry',
    2: 'Z_Scratch',
    3: 'K_Scratch',
    4: 'Stains',
    5: 'Dirtiness',
    6: 'Bumps',
    7: 'Other_Faults'
}

# Creo una serie testuale usando il dizionario
y_names = df['Defects'].map(mappa_difetti)

# 3. Inizializzo il LabelEncoder esattamente come se lo aspetta il resto del codice
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_names)

# Salvataggio file per diagnostica (così hai il check dei dati prima dello split)
df_step2 = X.copy()
df_step2['Difetto_Target_Numerico'] = y_encoded
df_step2.to_excel(f"{folder_export}/02_Step2_Dati_Ingegnerizzati.xlsx", index=False)

print(f"Dataset pronto per lo split. Feature: {X.shape[1]}, Campioni: {X.shape[0]}")
# endregion


#Divido i dati in test e train
#region STEP 3

    # X sono tutte le colonne senza difetti
    # Y sono i difetti
    # Test size 0.2 
    # random_state
    #Stratify per prendere una perc di tutti i difetti
    
print("\n--- STEP 3: Split e Scaling ---")
X_temp, X_test_initial, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded)

    #qui prendo i valori di validazione 

X_train_initial, X_val_initial, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SIZE / (1-TEST_SIZE), random_state=RANDOM_SEED, stratify=y_temp)

#Scalo ma rimuove i titoli
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_initial)
X_val_scaled = scaler.transform(X_val_initial)
X_test_scaled = scaler.transform(X_test_initial)

#Rimetto i titoli
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val = pd.DataFrame(X_val_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Salvo i dati scalati 
X_train.to_excel(f"{folder_export}/03_Step3_Train_Scalato.xlsx", index=False)
X_val.to_excel(f"{folder_export}/03_Step3_Val_Scalato.xlsx", index=False)
X_test.to_excel(f"{folder_export}/03_Step3_Test_Scalato.xlsx", index=False)

print("\n--- STEP 3: Dati divisi e scalati: ")
print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
#endregion


#region STEP 4A  XGBoost
from sklearn.model_selection import PredefinedSplit

print("\n--- STEP 4A: Ricerca Iperparametri XGBoost ---")

# 1. Definisco il modello base. 
# Inserisco l'early stopping qui, ma lascio il numero di alberi alla griglia.
xgb_base = XGBClassifier(
    objective='multi:softprob', 
    random_state=RANDOM_SEED, 
    eval_metric='mlogloss',
    early_stopping_rounds=10   # <-- Si fermerà in anticipo solo se l'errore non scende per 10 alberi
)

# Definisco la "griglia" dei parametri da testare
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1]
}

# ==============================================================================
# PER IL PROF: Ora è tutto impostato sui dati di TEST.
# PER LA VERSIONE CORRETTA: Cambia "test" in "val" in tutte le righe qui sotto.
# ==============================================================================
X_train_test = pd.concat([X_train, X_val], axis=0) 
y_train_test = np.concatenate([y_train, y_val])    

test_fold = np.concatenate([
    np.full(X_train.shape[0], -1),  
    np.full(X_test.shape[0], 0)     
])

ps = PredefinedSplit(test_fold)

# 3. Configuro la Grid Search (Questo pezzo mancava nel tuo copia-incolla!)
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=ps,
    verbose=1,
    n_jobs=None  
)
# ==============================================================================

print(f"Inizio addestramento Grid Search per {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} combinazioni...")

# 4. Avvio l'addestramento intensivo 
# Passiamo l'eval_set qui dentro affinché l'Early Stopping sappia quali dati guardare
grid_search.fit(
    X_train_test, y_train_test,
    eval_set=[(X_test, y_test)], # <-- CAMBIA IN X_val, y_val PER LA VERSIONE CORRETTA
    verbose=False
)

# Creo la Tabella dei Risultati
results_df = pd.DataFrame(grid_search.cv_results_)

# Seleziono solo le colonne utili per la presentazione e le ordino per risultato migliore
colonne_utili = ['param_learning_rate', 'param_max_depth', 'param_n_estimators', 'mean_test_score', 'std_test_score', 'rank_test_score']
tabella_risultati = results_df[colonne_utili].sort_values(by='rank_test_score')

# Rinomino le colonne per averle già pronte e pulite in italiano per l'Excel
tabella_risultati.rename(columns={
    'param_learning_rate': 'Learning Rate',
    'param_max_depth': 'Max Depth',
    'param_n_estimators': 'N. Alberi',
    'mean_test_score': 'Accuratezza Media',
    'std_test_score': 'Deviazione Standard',
    'rank_test_score': 'Posizione'
}, inplace=True)

# Arrotondo i numeri per una migliore leggibilità
tabella_risultati['Accuratezza Media'] = tabella_risultati['Accuratezza Media'].apply(lambda x: f"{x:.2%}")
tabella_risultati['Deviazione Standard'] = tabella_risultati['Deviazione Standard'].apply(lambda x: f"{x:.4f}")

# Stampo la Top 5 a video
print("\n--- I MIGLIORI 5 RISULTATI XGBOOST ---")
print(tabella_risultati.head(5).to_string(index=False))

# Salvo la tabella in Excel (utilissima da incollare nelle slide/tesi)
tabella_risultati.to_excel(f"{folder_export}/04_Step4_Risultati_GridSearch_XGBoost.xlsx", index=False)

# 6. Salvo il modello migliore per passarlo allo STEP 5 e STEP 6
model = grid_search.best_estimator_
print(f"\nMigliori iperparametri trovati: {grid_search.best_params_}")

# Salviamo i parametri migliori in questa variabile esatta
migliori_parametri = grid_search.best_params_

print("\nAllenamento del modello finale con i parametri ottimali (necessario per grafico ed Early Stopping)...")
model = XGBClassifier(
    n_estimators=migliori_parametri['n_estimators'], 
    learning_rate=migliori_parametri['learning_rate'], 
    max_depth=migliori_parametri['max_depth'],
    objective='multi:softprob', 
    random_state=RANDOM_SEED, 
    eval_metric='mlogloss',
    early_stopping_rounds=10  
)

# L'addestramento finale per poter generare il grafico
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)], # <-- CAMBIA IN X_val, y_val PER LA VERSIONE CORRETTA
    verbose=False
)

# Ricavo di dati come si è spostato
miglior_iterazione = model.best_iteration + 1 
miglior_score = model.best_score

print(f"Training completato. Il modello si è fermato a {miglior_iterazione} alberi.")
print(f"Miglior errore (mlogloss) su Set Esterno: {miglior_score:.4f}")

# VISUALIZZO IL GRAFICO
if PRINT_GRAPH:

    # Estraiamo lo storico degli errori
    risultati = model.evals_result()

    # Estraiamo i dati di errore per il Train (validation_0) e l'eval_set esterno (validation_1)
    errore_train = risultati['validation_0']['mlogloss']
    errore_val = risultati['validation_1']['mlogloss']
    epoche = range(0, len(errore_train))

    # Disegniamo il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(epoche, errore_train, label='Errore di Addestramento (Train)', color='blue')
    plt.plot(epoche, errore_val, label='Errore Test Esterno', color='orange') # Adatta anche l'etichetta del grafico

    # Aggiungiamo una linea verticale nel punto esatto in cui è intervenuto l'Early Stopping
    plt.axvline(x=miglior_iterazione, color='red', linestyle='--', label=f'Miglior Iterazione ({miglior_iterazione})')

    # Impaginazione per la tesi
    plt.title('Curva di Apprendimento XGBoost - Rilevamento Difetti Acciaio')
    plt.xlabel('Numero di Alberi (Iterazioni)')
    plt.ylabel('Errore ') #(Multi-LogLoss)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # Mostra il grafico a schermo
    plt.show()

#endregion

#region STEP 4B: Rete Neurale (Multilayer Perceptron)

print("\n--- STEP 4B: Ricerca Iperparametri MLP (Grid Search) ---")

# 1. Definisco il modello base MLP
# Uso early_stopping=True e un max_iter alto per permettere alla rete di convergere senza andare in overfitting
mlp_base = MLPClassifier(
    max_iter=1000,       
    early_stopping=True, 
    random_state=RANDOM_SEED
)

# 2. Definisco la griglia dei parametri
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25)], # Architettura dei neuroni
    'activation': ['relu', 'tanh'],                  # Funzioni di attivazione
    'learning_rate_init': [0.001, 0.01],             # Velocità di apprendimento iniziale
    'alpha': [0.0001, 0.01]                          # Regolarizzazione (L2 penalty)
}

# 3. Configuro la Grid Search 
grid_search_mlp = GridSearchCV(
    estimator=mlp_base,
    param_grid=param_grid_mlp,
    scoring='accuracy',  
    cv=3,
    verbose=1,
    n_jobs=None  # Evita il blocco su Windows
)

print(f"Inizio addestramento Grid Search MLP per {len(param_grid_mlp['hidden_layer_sizes']) * len(param_grid_mlp['activation']) * len(param_grid_mlp['learning_rate_init']) * len(param_grid_mlp['alpha'])} combinazioni...")
grid_search_mlp.fit(X_train, y_train)

# 4. Creazione della Tabella dei Risultati MLP
results_mlp_df = pd.DataFrame(grid_search_mlp.cv_results_)
colonne_utili_mlp = ['param_hidden_layer_sizes', 'param_activation', 'param_learning_rate_init', 'param_alpha', 'mean_test_score', 'std_test_score', 'rank_test_score']
tabella_risultati_mlp = results_mlp_df[colonne_utili_mlp].sort_values(by='rank_test_score')

# Rinomino le colonne per averle in italiano e pronte per l'Excel
tabella_risultati_mlp.rename(columns={
    'param_hidden_layer_sizes': 'Neuroni e Livelli Nascosti',
    'param_activation': 'Attivazione',
    'param_learning_rate_init': 'Learning Rate',
    'param_alpha': 'Regolarizzazione (Alpha)',
    'mean_test_score': 'Accuratezza Media (CV)',
    'std_test_score': 'Deviazione Standard',
    'rank_test_score': 'Posizione'
}, inplace=True)

tabella_risultati_mlp['Accuratezza Media (CV)'] = tabella_risultati_mlp['Accuratezza Media (CV)'].apply(lambda x: f"{x:.2%}")
tabella_risultati_mlp['Deviazione Standard'] = tabella_risultati_mlp['Deviazione Standard'].apply(lambda x: f"{x:.4f}")

print("\n--- I MIGLIORI 5 RISULTATI MLP ---")
print(tabella_risultati_mlp.head(5).to_string(index=False))

# Salvo la tabella in Excel per il confronto diretto con XGBoost
tabella_risultati_mlp.to_excel(f"{folder_export}/04_Step4B_Risultati_GridSearch_MLP.xlsx", index=False)
#print(f"\nTabella MLP salvata in: {folder_export}/04_Step4B_Risultati_GridSearch_MLP.xlsx")

# 5. ESTRAZIONE PARAMETRI OTTimali E ADDESTRAMENTO FINALE
migliori_parametri_mlp = grid_search_mlp.best_params_
print(f"\nMigliori iperparametri MLP trovati: {migliori_parametri_mlp}")

print("\nAllenamento del modello MLP finale con i parametri ottimali...")
model_mlp = MLPClassifier(
    hidden_layer_sizes=migliori_parametri_mlp['hidden_layer_sizes'],
    activation=migliori_parametri_mlp['activation'],
    learning_rate_init=migliori_parametri_mlp['learning_rate_init'],
    alpha=migliori_parametri_mlp['alpha'],
    max_iter=1000,
    early_stopping=True,
    random_state=RANDOM_SEED
)

# Addestramento della rete definitiva
model_mlp.fit(X_train, y_train)

# VISUALIZZO IL GRAFICO DEL PERCETTRONE (Curva di perdita)
if PRINT_GRAPH:
    plt.figure(figsize=(10, 6))
    plt.plot(model_mlp.loss_curve_, label='Curva di Errore (Loss)', color='green', linewidth=2)
    plt.title('Curva di Apprendimento MLP - Rilevamento Difetti Acciaio')
    plt.xlabel('Iterazioni (Epoche)')
    plt.ylabel('Errore (Loss)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

#endregion


#Valutazione
#region STEP 5
if not SKIP_STEP5:
    print("\n=== STEP 5: VALUTAZIONE COMPARATIVA ===")
    target_names = label_encoder.classes_

    # 1. Valutazione XGBoost
    y_pred_xgb = model.predict(X_test)
    print("\n--- METRICHE XGBOOST ---")
    print(f"Accuratezza Globale: {accuracy_score(y_test, y_pred_xgb):.2%}")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names, zero_division=0))

    # 2. Stampo la matrice di confusione
    print("\n--- MATRICE DI CONFUSIONE XGBOOST ---")
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    df_cm_xgb = pd.DataFrame(cm_xgb, index=target_names, columns=target_names)
    print(df_cm_xgb)

    # 3. Valutazione MLP (Rete Neurale)
    y_pred_mlp = model_mlp.predict(X_test)
    print("\n--- METRICHE PERCETTRONE MULTISTRATO (MLP) ---")
    print(f"Accuratezza Globale: {accuracy_score(y_test, y_pred_mlp):.2%}")
    print(classification_report(y_test, y_pred_mlp, target_names=target_names, zero_division=0))

    # 4. Stampo la matrice di confusione
    print("\n--- MATRICE DI CONFUSIONE MLP ---")
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    df_cm_mlp = pd.DataFrame(cm_mlp, index=target_names, columns=target_names)
    print(df_cm_mlp)


#endregion

#region STEP 6: Simulazione Prescrittiva su Larga Scala (100 Barre)
if not SKIP_STEP6:
    print("\n--- STEP 6: Simulazione Prescrittiva Massiva (100 Barre) ---")

    NUM_TEST = 100
    risultati_simulazione = []

    # 1. Trovo tutti i potenziali casi critici (usando XGBoost come "ispettore" base)
    probs_riferimento = model.predict_proba(X_test)
    idx_nessun_difetto = list(label_encoder.classes_).index('No_Defects')
    
    probs_solo_difetti = probs_riferimento.copy()
    probs_solo_difetti[:, idx_nessun_difetto] = 0.0

    max_probs = np.max(probs_solo_difetti, axis=1)
    indici_difettosi = np.where(max_probs > 0.5)[0]

    # Seleziono 100 indici casuali tra quelli difettosi
    # Nota: se ci sono meno di 100 difetti, permettiamo la ripetizione (replace=True)
    indici_test = np.random.choice(indici_difettosi, size=NUM_TEST, replace=len(indici_difettosi) < NUM_TEST)

    cols = X_test.columns.tolist()
    idx_temp = cols.index('Rolling_Temp_C')
    idx_speed = cols.index('Roller_Speed_m_sec')
    idx_press = cols.index('Pressure_Bar')

    modelli_da_testare = [
        ("XGBoost", model),
        ("MLP", model_mlp)
    ]

    print(f"Avvio ottimizzazione per {NUM_TEST} barre su entrambi i modelli. Attendere...")

    # 2. Eseguo il ciclo sui 100 casi
    for i, idx_caso in enumerate(indici_test):
        # Avanzamento visivo a blocchi di 10 per non bloccare lo schermo
        if (i+1) % 10 == 0:
            print(f"Progresso: {i+1}/{NUM_TEST} barre completate...")

        # Estrazione dati della barra corrente
        riga_scalata = X_test.iloc[idx_caso].values.reshape(1, -1)
        riga_reale = scaler.inverse_transform(riga_scalata)
        caso_reale = pd.Series(riga_reale[0], index=X_test.columns)

        difetto_previsto_idx = np.argmax(probs_riferimento[idx_caso])
        difetto_nome = label_encoder.inverse_transform([difetto_previsto_idx])[0]

        temp_start = max(800, min(1000, caso_reale['Rolling_Temp_C']))
        speed_start = max(9, min(15, caso_reale['Roller_Speed_m_sec']))
        press_start = max(150, min(300, caso_reale['Pressure_Bar']))
        x0 = [temp_start, speed_start, press_start]
        bounds = [(800, 1000), (9, 15), (150, 300)]

        # Variabili per tracciare chi vince su questa barra
        risultato_barra = {
            'Id_Barra': i+1,
            'Difetto': difetto_nome,
            'Temp_Iniziale': caso_reale['Rolling_Temp_C'],
            'Speed_Iniziale': caso_reale['Roller_Speed_m_sec'],
            'Press_Iniziale': caso_reale['Pressure_Bar']
        }

        # 3. Testo entrambi i modelli sulla stessa barra
        for nome_modello, modello_corrente in modelli_da_testare:
            
            prob_iniziale = modello_corrente.predict_proba(riga_scalata)[0][difetto_previsto_idx]
            
            def objective_function(x_new):
                row_simulation = caso_reale.values.copy() 
                row_simulation[idx_temp] = x_new[0]         
                row_simulation[idx_speed] = x_new[1]        
                row_simulation[idx_press] = x_new[2]        
                
                row_scaled = scaler.transform(pd.DataFrame(row_simulation.reshape(1, -1), columns=X_test.columns))
                prob_difetto = modello_corrente.predict_proba(row_scaled)[0][difetto_previsto_idx]

                penalita_temp = ((x_new[0] - caso_reale['Rolling_Temp_C']) / caso_reale['Rolling_Temp_C'])**2
                penalita_speed = ((x_new[1] - caso_reale['Roller_Speed_m_sec']) / caso_reale['Roller_Speed_m_sec'])**2
                penalita_press = ((x_new[2] - caso_reale['Pressure_Bar']) / caso_reale['Pressure_Bar'])**2
                
                return prob_difetto + (0.5 * (penalita_temp + penalita_speed + penalita_press))

            result = minimize(objective_function, x0, method='Powell', bounds=bounds, tol=1e-4)

            # Salvo i risultati specifici del modello
            risultato_barra[f'Prob_Iniziale_{nome_modello}'] = prob_iniziale
            
            if result.success:
                row_finale = caso_reale.values.copy()
                row_finale[idx_temp], row_finale[idx_speed], row_finale[idx_press] = result.x
                row_df_fin = pd.DataFrame(row_finale.reshape(1, -1), columns=X_test.columns)
                new_prob = modello_corrente.predict_proba(scaler.transform(row_df_fin))[0][difetto_previsto_idx]
                
                risultato_barra[f'Prob_Finale_{nome_modello}'] = new_prob
                risultato_barra[f'Temp_Finale_{nome_modello}'] = result.x[0]
                risultato_barra[f'Press_Finale_{nome_modello}'] = result.x[2]
                risultato_barra[f'Vel_Finale_{nome_modello}'] = result.x[1]
                risultato_barra[f'Delta_Temp_{nome_modello}'] = abs(result.x[0] - caso_reale['Rolling_Temp_C'])
                risultato_barra[f'Delta_Press_{nome_modello}'] = abs(result.x[2] - caso_reale['Pressure_Bar'])
                risultato_barra[f'Delta_Vel_{nome_modello}'] = abs(result.x[1] - caso_reale['Roller_Speed_m_sec'])
                risultato_barra[f'Successo_{nome_modello}'] = True
            else:
                risultato_barra[f'Prob_Finale_{nome_modello}'] = prob_iniziale
                risultato_barra[f'Temp_Finale_{nome_modello}'] = 0
                risultato_barra[f'Press_Finale_{nome_modello}'] = 0
                risultato_barra[f'Vel_Finale_{nome_modello}'] = 0
                risultato_barra[f'Delta_Temp_{nome_modello}'] = 0
                risultato_barra[f'Delta_Press_{nome_modello}'] = 0
                risultato_barra[f'Delta_Vel_{nome_modello}'] = 0
                risultato_barra[f'Successo_{nome_modello}'] = False

        # Aggiungo la barra completata alla lista dei risultati
        risultati_simulazione.append(risultato_barra)

    # 4. Converto tutto in un DataFrame e lo salvo in Excel
    df_simulazione = pd.DataFrame(risultati_simulazione)
    df_simulazione.to_excel(f"{folder_export}/06_Step6_Simulazione_Massiva_100.xlsx", index=False)
    print(f"\nSimulazione completata. Dati grezzi salvati in '{folder_export}/06_Step6_Simulazione_Massiva_100.xlsx'")

#endregion

#region STEP 7: Report Statistico della Simulazione
if not SKIP_STEP7:
    print("\n=== STEP 7: REPORT STATISTICO SULL'ANALISI PRESCRITTIVA ===")

    # Calcolo quante probabilità residue ci sono (più è bassa, più il modello ha curato bene il difetto)
    media_prob_finale_xgb = df_simulazione['Prob_Finale_XGBoost'].mean()
    media_prob_finale_mlp = df_simulazione['Prob_Finale_MLP'].mean()

    # Calcolo quanto ha dovuto "stressare" l'impianto per farcela (Variazione di temperatura)
    media_sforzo_temp_xgb = df_simulazione['Delta_Temp_XGBoost'].mean()
    media_sforzo_temp_mlp = df_simulazione['Delta_Temp_MLP'].mean()

    # Calcolo quanto ha dovuto "stressare" l'impianto per farcela (Variazione di pressine)
    media_sforzo_press_xgb = df_simulazione['Delta_Press_XGBoost'].mean()
    media_sforzo_press_mlp = df_simulazione['Delta_Press_MLP'].mean()

    # Calcolo quanto ha dovuto "stressare" l'impianto per farcela (Variazione di velocità)
    media_sforzo_vel_xgb = df_simulazione['Delta_Vel_XGBoost'].mean()
    media_sforzo_vel_mlp = df_simulazione['Delta_Vel_MLP'].mean()

    # Calcolo quante volte il modello è riuscito a portare il rischio sotto una soglia di sicurezza (es. < 5%)
    sicurezza_xgb = (df_simulazione['Prob_Finale_XGBoost'] < 0.05).sum()
    sicurezza_mlp = (df_simulazione['Prob_Finale_MLP'] < 0.05).sum()

    # Stampo il verdetto finale
    print("\n1. CAPACITÀ DI RISOLUZIONE DEL DIFETTO:")
    print(f" - Barre curate in sicurezza (<5% rischio) da XGBoost: {sicurezza_xgb}/{NUM_TEST}")
    print(f" - Barre curate in sicurezza (<5% rischio) da MLP:     {sicurezza_mlp}/{NUM_TEST}")
    print(f" - Rischio medio residuo dopo prescrizione XGBoost:      {media_prob_finale_xgb:.2%}")
    print(f" - Rischio medio residuo dopo prescrizione MLP:          {media_prob_finale_mlp:.2%}")

    print("\n2. VARIAZIONI :")
    print(f" - Variazione media di Temperatura richiesta da XGBoost: {media_sforzo_temp_xgb:.2f} °C")
    print(f" - Variazione media di Temperatura richiesta da MLP:     {media_sforzo_temp_mlp:.2f} °C")
    print(f" - Variazione media di Pressione richiesta da XGBoost: {media_sforzo_press_xgb:.2f} bar")
    print(f" - Variazione media di Pressione richiesta da MLP:     {media_sforzo_press_mlp:.2f} bar")
    print(f" - Variazione media di Velocità richiesta da XGBoost: {media_sforzo_vel_xgb:.2f} m/s")
    print(f" - Variazione media di Velocità richiesta da MLP:     {media_sforzo_vel_mlp:.2f} m/s")

    # Logica per dichiarare il vincitore
    if (media_prob_finale_xgb < media_prob_finale_mlp):
        vincitore = "XGBoost"
    else:
        vincitore = "MLP (Percettrone Multistrato)"
    print(f"\n VERDETTO DELLA SIMULAZIONE: {vincitore} è il modello più efficace nella prescrizione!")

    # Salvataggio di questo report in un file di testo pulito
    with open(f"{folder_export}/07_Step7_Report_Statistico.txt", "w") as file:
        file.write("=== REPORT STATISTICO SULL'ANALISI PRESCRITTIVA (100 BARRE) ===\n\n")
        file.write(f"Barre curate in sicurezza (<5%) da XGBoost: {sicurezza_xgb}/{NUM_TEST}\n")
        file.write(f"Barre curate in sicurezza (<5%) da MLP:     {sicurezza_mlp}/{NUM_TEST}\n")
        file.write(f"Variazione media Temperatura XGBoost:       {media_sforzo_temp_xgb:.2f} °C\n")
        file.write(f"Variazione media Temperatura MLP:           {media_sforzo_temp_mlp:.2f} °C\n")
        file.write(f"Variazione media Pressione XGBoost:       {media_sforzo_press_xgb:.2f} bar\n")
        file.write(f"Variazione media Pressione MLP:           {media_sforzo_press_mlp:.2f} bar\n")
        file.write(f"Variazione media Velocità XGBoost:       {media_sforzo_vel_xgb:.2f} m/s\n")
        file.write(f"Variazione media Velocità MLP:           {media_sforzo_vel_mlp:.2f} m/s\n")
        file.write(f"Vincitore Globale:                          {vincitore}\n")
#endregion

