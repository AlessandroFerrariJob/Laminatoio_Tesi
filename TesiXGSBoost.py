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
PRINT_GRAPH = True  #Visualizzo il grafico dell'errore

#varie utility
# region STEP 0

# Creo una cartella per tenere tutto in ordine
os.makedirs(folder_export, exist_ok=True)
# endregion


#Leggo il file excel, aggiungo i campi di temperatura velocità e pressione modificati in base al difetto
# region STEP 1 

print("--- STEP 1: Caricamento da file locale ({percorso_file}) ---")
df = pd.read_excel(path_file)

#  Aggiungo le colonne base di temperatura velocità e pressione
n_rows = len(df)
df['Rolling_Temp_C'] = np.random.normal(loc=950, scale=20, size=n_rows)
df['Roller_Speed_m_sec'] = np.random.normal(loc=10, scale=1, size=n_rows)
df['Pressure_Bar'] = np.random.normal(loc=200, scale=10, size=n_rows)

#difetto di sfogliatura, superificiale e altro diminuisco la temperatura
mask_defect = (df['Pastry'] == 1) | (df['Stains'] == 1)| (df['Other_Faults'] == 1)
df.loc[mask_defect, 'Rolling_Temp_C'] -= np.random.uniform(50, 100, size=mask_defect.sum())

#difetto di graffi e altro aumento la velocità
mask_defect = (df['Z_Scratch'] == 1) | (df['K_Scratch'] == 1) | (df['Other_Faults'] == 1)
df.loc[mask_defect, 'Roller_Speed_m_sec'] += np.random.uniform(3, 6, size=mask_defect.sum())

#difetto sporco o irregolarità aumento la pressione
mask_defect = (df['Bumps'] == 1) | (df['Dirtiness'] == 1)
df.loc[mask_defect, 'Pressure_Bar'] += np.random.uniform(40, 80, size=mask_defect.sum())

#Salvo il file per diagnostica
df.to_excel(f"{folder_export}/01_Step1_Dati_Aumentati.xlsx", index=False)

print("--- STEP 1 {n_rows} righe trovate. Variabili simulate aggiunte.")
# endregion


#Tolgo le 7 colonne dei difetti e ne metto una numerica con il numero di difetto 
# region STEP 2 
print("\n--- STEP 2: Sistemazione colonna difetti ---")

# Nome delle colonne da cancellare
fault_columns = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

#Cancello le 7 colonne dei difetti 
X = df.drop(columns=fault_columns)

#Metto il nome dei difetti nell'array y_names
y_names = df[fault_columns].idxmax(axis=1)

# Assegno l'etichetta "Nessun_Difetto" alle righe senza difetto
mask_nessun_difetto = df[fault_columns].sum(axis=1) == 0
y_names.loc[mask_nessun_difetto] = 'No_Defects'

#Metto i numeri da 0 a  n al posto del nome dei difetti 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_names)

#Salvataggio file per diagnostica
df_step2 = X.copy()
df_step2['Difetto_Target_Numerico6'] = y_encoded
df_step2.to_excel(f"{folder_export}/02_Step2_Dati_Ingegnerizzati.xlsx", index=False)

print(f"Dataset pronto. Feature: {X.shape[1]}, Campioni: {X.shape[0]}")

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
X_test.to_excel(f"{folder_export}/03_Step3_Test_Scalato.xlsx", index=False)

print("\n--- STEP 3: Dati divisi e scalati: ")
print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
#endregion


#Training
#region STEP 4

print("\n--- STEP 4: Training  ---")

model = XGBClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=6,
    objective='multi:softprob', 
    random_state=RANDOM_SEED, 
    eval_metric='mlogloss',
    early_stopping_rounds=10  # SE l'errore sulla Validazione non scende per 10 cicli di fila
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)], # In XGBoost, il primo è validation_0, il secondo è validation_1
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


#Valutazione
#region STEP 5
if not SKIP_STEP5:
    print("\n--- STEP 5: Valutazione ---")
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_

    print(f"Accuratezza: {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

#endregion


# Cerco un valore con difetto e vedo quali sono i parametri da modifcare 
#region STEP 6
if not SKIP_STEP6:
    print("\n--- STEP 6: Analisi prescrittiva  ---")

    # Trovo il caso critico
    probs = model.predict_proba(X_test)

    # Ignoro i pezzi perfetti!
    # Troviamo l'indice (il numero) della categoria 'Nessun_Difetto'
    idx_nessun_difetto = list(label_encoder.classes_).index('No_Defects')
    # Creo una copia dell'array probs e azzero le probabilità dei  pezzi sani
    probs_solo_difetti = probs.copy()
    probs_solo_difetti[:, idx_nessun_difetto] = 0.0

    # Cerco il difetto con probabilità più alta
    max_probs = np.max(probs_solo_difetti, axis=1)
    worst_case_idx = np.argmax(max_probs)

    #Scalo il valore trovato
    riga_scalata = X_test.iloc[worst_case_idx].values.reshape(1, -1)
    riga_reale = scaler.inverse_transform(riga_scalata)
    caso_reale = pd.Series(riga_reale[0], index=X_test.columns)

    difetto_previsto_idx = np.argmax(probs[worst_case_idx])
    difetto_nome = label_encoder.inverse_transform([difetto_previsto_idx])[0]
    prob_iniziale = probs[worst_case_idx][difetto_previsto_idx]

    #Stampo a video il valore trovato
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


    #Creo una funzione per essere utilizzato dall'ottimizzatore 
    def objective_function(x_new):
        row_simulation = caso_reale.values.copy()   # Copio tutte le variabili
        row_simulation[idx_temp] = x_new[0]         # Metto nella parte di simulazioni i dati di tentativo 
        row_simulation[idx_speed] = x_new[1]        #
        row_simulation[idx_press] = x_new[2]        #
        # --- Ricreiamo il DataFrame con i nomi delle colonne ---
        row_df = pd.DataFrame(row_simulation.reshape(1, -1), columns=X_test.columns)
        # Scalo i valori
        row_scaled = scaler.transform(row_df)
        # Calcolo la probabilità di avere i difetti
        prediction = model.predict_proba(row_scaled)
        return prediction[0][difetto_previsto_idx]
    # fine della funzione 

    #Primo tentativo
    #bounds = [(800, 1100), (5, 20), (150, 300)]
    #Imposto i limiti
    bounds = [(900, 1000), (9, 15), (150, 250)]


    print("\nAvvio ottimizzazione matematica...")

    x0 = [caso_reale['Rolling_Temp_C'], 
            caso_reale['Roller_Speed_m_sec'], 
            caso_reale['Pressure_Bar']]

    #Algoritmo di ottimizzazione
    #Questo non funzionava per via del problòema della derivata
    #result = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds, tol=1e-4)
    result = minimize(objective_function, x0, method='Powell', bounds=bounds, tol=1e-4)

    #Stampo a video il risultato ottenuto 
    if result.success:
        new_temp, new_speed, new_press = result.x
        new_prob = result.fun
        
        print("\n--- PRESCRIZIONE OPERATIVA ---")
        print(f"Per ridurre il rischio '{difetto_nome}':")
        print(f"1. Temperatura: {new_temp:.2f} °C (era {x0[0]:.2f})")
        print(f"2. Velocità:    {new_speed:.2f} m/s (era {x0[1]:.2f})")
        print(f"3. Pressione:   {new_press:.2f} Bar (era {x0[2]:.2f})")
        print(f"\nRischio ridotto dal {prob_iniziale:.2%} al {new_prob:.2%}")
    else:
        print("Ottimizzazione fallita:", result.message)
#endregion


