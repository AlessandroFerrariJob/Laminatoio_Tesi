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
PRINT_GRAPH = False  #Visualizzo il grafico dell'errore

# endregion

#varie utility
# region STEP 0

# Creo una cartella per tenere tutto in ordine
os.makedirs(folder_export, exist_ok=True)
# endregion


#Leggo il file excel, aggiungo i campi di temperatura velocità e pressione modificati in base al difetto
# region STEP 1 

N_RIGHE_TARGET = 10000

#print("--- STEP 1: Caricamento da file locale ) ---")
df = pd.DataFrame(index=range(N_RIGHE_TARGET))

# Definisco i numeri possibili (da 0 a 7) e la probabilità
valori_possibili = [0, 1, 2, 3, 4, 5, 6, 7]
probabilita = [0.9, 0.02, 0.02, 0.01, 0.01, 0.01, 0.015, 0.015]


#  Aggiungo le colonne base di temperatura velocità e pressione
n_rows = N_RIGHE_TARGET
df['Rolling_Temp_C'] = np.random.normal(loc=950, scale=20, size=n_rows)
df['Roller_Speed_m_sec'] = np.random.normal(loc=10, scale=1, size=n_rows)
df['Pressure_Bar'] = np.random.normal(loc=200, scale=10, size=n_rows)
df['Defects'] = np.random.choice(valori_possibili, size=n_rows, p=probabilita)

# 'No_Defects'  - 1 'Pastry', 2 'Z_Scratch', 3 'K_Scratch', 4 'Stains',5 'Dirtiness', 6'Bumps', 7 'Other_Faults' 
#difetto di sfogliatura, superificiale e altro diminuisco la temperatura (Pastry,Stains,Other_Faults)
mask_defect = (df['Defects'] == 1) | (df['Defects'] == 4)| (df['Defects'] == 7)
df.loc[mask_defect, 'Rolling_Temp_C'] -= np.random.uniform(50, 100, size=mask_defect.sum())

#difetto di graffi e altro aumento la velocità (Z_Scratch,K_Scratch,Other_Faults)
mask_defect = (df['Defects'] == 3) | (df['Defects'] == 4) | (df['Defects'] == 7)
df.loc[mask_defect, 'Roller_Speed_m_sec'] += np.random.uniform(3, 6, size=mask_defect.sum())

#difetto sporco o irregolarità aumento la pressione(Bumps,Dirtiness)
mask_defect = (df['Defects'] == 5) | (df['Defects'] == 6)
df.loc[mask_defect, 'Pressure_Bar'] += np.random.uniform(40, 80, size=mask_defect.sum())

#Salvo il file per diagnostica
df.to_excel(f"{folder_export}/01_Step1_Dati_Aumentati.xlsx", index=False)
print("--- STEP 1 {n_rows} righe trovate. Variabili simulate aggiunte.")
# endregion

#step 2 alternativo

df.to_excel(f"{folder_export}/02_Step2_Dati_Ingegnerizzati.xlsx", index=False)

X = df
fault_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
y_names = df[fault_columns].idxmax(axis=1)
mask_nessun_difetto = df[fault_columns].sum(axis=1) == 0
y_names.loc[mask_nessun_difetto] = 'No_Defects'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_names)


# #Tolgo le 7 colonne dei difetti e ne metto una numerica con il numero di difetto 
# # region STEP 2 
# print("\n--- STEP 2: Sistemazione colonna difetti ---")

# # Nome delle colonne da cancellare
# fault_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# #Cancello le 7 colonne dei difetti 
# X = df.drop(columns=fault_columns)

# #Metto il nome dei difetti nell'array y_names
# y_names = df[fault_columns].idxmax(axis=1)

# # Assegno l'etichetta "Nessun_Difetto" alle righe senza difetto
# mask_nessun_difetto = df[fault_columns].sum(axis=1) == 0
# y_names.loc[mask_nessun_difetto] = 'No_Defects'

# #Metto i numeri da 0 a  n al posto del nome dei difetti 
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y_names)

# #Salvataggio file per diagnostica
# df_step2 = X.copy()
# df_step2['Difetto_Target_Numerico6'] = y_encoded
# df_step2.to_excel(f"{folder_export}/02_Step2_Dati_Ingegnerizzati.xlsx", index=False)

# print(f"Dataset pronto. Feature: {X.shape[1]}, Campioni: {X.shape[0]}")

# # endregion


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
X_test.to_excel(f"{folder_export}/03_Step3_Test_Scalato.xlsx", index=False)

print("\n--- STEP 3: Dati divisi e scalati: ")
print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
#endregion


#Training
#region STEP 4A  XGBoost

print("\n--- STEP 4: Ricerca Iperparametri XGBoost  ---")

# Definisco il modello base XGBoost
xgb_base = XGBClassifier(
    objective='multi:softprob', 
    random_state=RANDOM_SEED, 
    eval_metric='mlogloss'
)

# Definisco la "griglia" dei parametri da testare
# ---------------    VERIFICARE SE CAMBIARE I VALORI     ----------------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Configuro la Grid Search (CV=3 significa che fa una validazione incrociata a 3 fold)
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='accuracy',  # Ottimizziamo per accuratezza generale
    cv=3,
    verbose=1, # Stampa a video l'avanzamento
    n_jobs=None  
)

print(f"Inizio addestramento Grid Search per {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} combinazioni...")

# Avvio l'addestramento intensivo 
grid_search.fit(X_train, y_train)

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
    'mean_test_score': 'Accuratezza Media (CV)',
    'std_test_score': 'Deviazione Standard',
    'rank_test_score': 'Posizione'
}, inplace=True)

# Arrotondo i numeri per una migliore leggibilità
tabella_risultati['Accuratezza Media (CV)'] = tabella_risultati['Accuratezza Media (CV)'].apply(lambda x: f"{x:.2%}")
tabella_risultati['Deviazione Standard'] = tabella_risultati['Deviazione Standard'].apply(lambda x: f"{x:.4f}")

# Stampo la Top 5 a video
print("\n--- I MIGLIORI 5 RISULTATI XGBOOST ---")
print(tabella_risultati.head(5).to_string(index=False))

# Salvo la tabella in Excel (utilissima da incollare nelle slide/tesi)
tabella_risultati.to_excel(f"{folder_export}/04_Step4_Risultati_GridSearch_XGBoost.xlsx", index=False)
print(f"\nTabella completa salvata in: {folder_export}/04_Step4_Risultati_GridSearch_XGBoost.xlsx")

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

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=False
)

# RIcavo di dati come si è spostato
miglior_iterazione = model.best_iteration + 1 
miglior_score = model.best_score

print(f"Training completato. Il modello si è fermato a {miglior_iterazione} alberi.")
print(f"Miglior errore (mlogloss) su Validazione: {miglior_score:.4f}")

# VISUALIZZO IL GRAFICO
if PRINT_GRAPH:

    # Estraiamo lo storico degli errori
    risultati = model.evals_result()

    # Estraiamo i dati di errore per il Train (validation_0) e la Validazione (validation_1)
    errore_train = risultati['validation_0']['mlogloss']
    errore_val = risultati['validation_1']['mlogloss']
    epoche = range(0, len(errore_train))

    # Disegniamo il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(epoche, errore_train, label='Errore di Addestramento (Train)', color='blue')
    plt.plot(epoche, errore_val, label='Errore di Validazione (Validation)', color='orange')

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
print(f"\nTabella MLP salvata in: {folder_export}/04_Step4B_Risultati_GridSearch_MLP.xlsx")

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
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))

    # 2. Valutazione MLP (Rete Neurale)
    y_pred_mlp = model_mlp.predict(X_test)
    print("\n--- METRICHE PERCETTRONE MULTISTRATO (MLP) ---")
    print(f"Accuratezza Globale: {accuracy_score(y_test, y_pred_mlp):.2%}")
    print(classification_report(y_test, y_pred_mlp, target_names=target_names))
#endregion


# Cerco un valore con difetto e vedo quali sono i parametri da modifcare 
#region STEP 6
if not SKIP_STEP6:
    print("\n--- STEP 6: Analisi Prescrittiva Comparativa ---")

    # Creiamo una lista contenente il nome e la variabile dei due modelli da testare
    modelli_da_testare = [
        ("XGBoost", model),
        ("Percettrone Multistrato (MLP)", model_mlp)
    ]

    # Il ciclo eseguirà tutta l'analisi prima per XGBoost e poi per l'MLP
    for nome_modello, modello_corrente in modelli_da_testare:
        
        print(f"\n{'='*50}")
        print(f" INIZIO OTTIMIZZAZIONE CON: {nome_modello}")
        print(f"{'='*50}")

        # Trovo il caso critico calcolato dal MODELLO CORRENTE
        probs = modello_corrente.predict_proba(X_test)

        # Ignoro i pezzi perfetti
        idx_nessun_difetto = list(label_encoder.classes_).index('No_Defects')
        probs_solo_difetti = probs.copy()
        probs_solo_difetti[:, idx_nessun_difetto] = 0.0

        # Cerco la probabilità massima di difetto per ogni pezzo
        max_probs = np.max(probs_solo_difetti, axis=1)

        # Trovo tutti gli indici dei pezzi che hanno un difetto evidente (probabilità > 50%)
        indici_difettosi = np.where(max_probs > 0.5)[0]

        # Controllo che ci sia almeno un pezzo difettoso
        if len(indici_difettosi) > 0:
            worst_case_idx = np.random.choice(indici_difettosi)
        else:
            worst_case_idx = np.argmax(max_probs)

        # Scalo il valore trovato
        riga_scalata = X_test.iloc[worst_case_idx].values.reshape(1, -1)
        riga_reale = scaler.inverse_transform(riga_scalata)
        caso_reale = pd.Series(riga_reale[0], index=X_test.columns)

        difetto_previsto_idx = np.argmax(probs[worst_case_idx])
        difetto_nome = label_encoder.inverse_transform([difetto_previsto_idx])[0]
        prob_iniziale = probs[worst_case_idx][difetto_previsto_idx]

        print(f"\nCASO IN ESAME (Riga {worst_case_idx}):")
        print(f"Difetto Rilevato: {difetto_nome} (Probabilità: {prob_iniziale:.2%})")
        print("Parametri Attuali (Reali):")
        print(f"- Temperatura: {caso_reale['Rolling_Temp_C']:.2f} °C")
        print(f"- Velocità:    {caso_reale['Roller_Speed_m_sec']:.2f} m/s")
        print(f"- Pressione:   {caso_reale['Pressure_Bar']:.2f} Bar")

        cols = X_test.columns.tolist()
        idx_temp = cols.index('Rolling_Temp_C')
        idx_speed = cols.index('Roller_Speed_m_sec')
        idx_press = cols.index('Pressure_Bar')

        # Funzione per l'ottimizzatore (usa dinamicamente il modello_corrente)
# Funzione per l'ottimizzatore (usa dinamicamente il modello_corrente)
        def objective_function(x_new):
            row_simulation = caso_reale.values.copy() 
            row_simulation[idx_temp] = x_new[0]         
            row_simulation[idx_speed] = x_new[1]        
            row_simulation[idx_press] = x_new[2]        
            
            row_df = pd.DataFrame(row_simulation.reshape(1, -1), columns=X_test.columns)
            row_scaled = scaler.transform(row_df)
            
            # 1. Calcolo la probabilità del difetto
            prediction = modello_corrente.predict_proba(row_scaled)
            prob_difetto = prediction[0][difetto_previsto_idx]

            # 2. Calcolo la penalità per l'eccessiva variazione dai parametri reali
            # Usiamo la variazione percentuale al quadrato per avere numeri gestibili
            penalita_temp = ((x_new[0] - caso_reale['Rolling_Temp_C']) / caso_reale['Rolling_Temp_C'])**2
            penalita_speed = ((x_new[1] - caso_reale['Roller_Speed_m_sec']) / caso_reale['Roller_Speed_m_sec'])**2
            penalita_press = ((x_new[2] - caso_reale['Pressure_Bar']) / caso_reale['Pressure_Bar'])**2
            
            somma_penalita = penalita_temp + penalita_speed + penalita_press

            # 3. Parametro Lambda: quanto "costa" muovere i parametri? 
            # Più è alto, più l'ottimizzatore cercherà valori vicini a quelli di partenza.
            # (Puoi variare questo 0.5 per vedere come cambia il comportamento)
            lambda_peso = 0.5 

            # L'ottimizzatore ora cerca un compromesso tra probabilità zero e parametri stabili
            return prob_difetto + (lambda_peso * somma_penalita)

        # Imposto i limiti fisici per l'impianto
        bounds = [(800, 1000), (9, 15), (150, 300)]

       # print("\nAvvio ottimizzazione matematica (Metodo: Powell)...")
        #x0 = [caso_reale['Rolling_Temp_C'], 
         #     caso_reale['Roller_Speed_m_sec'], 
          #    caso_reale['Pressure_Bar']]

        # Avvio ottimizzazione matematica...
        # Modifica per garantire che il punto di partenza sia dentro i limiti dell'impianto
        temp_start = max(800, min(1000, caso_reale['Rolling_Temp_C']))
        speed_start = max(9, min(15, caso_reale['Roller_Speed_m_sec']))
        press_start = max(150, min(300, caso_reale['Pressure_Bar']))

        x0 = [temp_start, speed_start, press_start]

        # Esecuzione dell'ottimizzazione
        result = minimize(objective_function, x0, method='Powell', bounds=bounds, tol=1e-4)

        if result.success:
            new_temp, new_speed, new_press = result.x
            #new_prob = result.fun
            # Ricalcolo la probabilità pura per la stampa a video (senza la penalità)
            row_finale = caso_reale.values.copy()
            row_finale[idx_temp], row_finale[idx_speed], row_finale[idx_press] = new_temp, new_speed, new_press
            row_df_fin = pd.DataFrame(row_finale.reshape(1, -1), columns=X_test.columns)
            new_prob = modello_corrente.predict_proba(scaler.transform(row_df_fin))[0][difetto_previsto_idx]
            
            print(f"\n--- PRESCRIZIONE OPERATIVA ({nome_modello}) ---")
            print(f"Per ridurre il rischio '{difetto_nome}':")
            print(f"1. Temperatura: {new_temp:.2f} °C (era {x0[0]:.2f})")
            print(f"2. Velocità:    {new_speed:.2f} m/s (era {x0[1]:.2f})")
            print(f"3. Pressione:   {new_press:.2f} Bar (era {x0[2]:.2f})")
            print(f"\nRischio ridotto dal {prob_iniziale:.2%} al {new_prob:.2%}")
        else:
            print("Ottimizzazione fallita:", result.message)

#endregion
