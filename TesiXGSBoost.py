"""
===============================================================================
Modulo:         TesiXGSBoost.py
Autore:         Alessandro Ferrari
Data:           08 agosto 2026
Versione:       2.1 (Integrazione TensorFlow)
===============================================================================

Descrizione:
Questo modulo contiene il programma di tesi di Alessandro Ferrari. 
Nel modulo viene fatta una sia un analisi predittiva e  che una prescrittiva 
confrontando due algoritmi: XGBoost e Percettrone multistrato.

Dipendenze:
    - Acquisizione e Manipolazione: pandas, numpy, ucimlrepo
    - Machine Learning Predittivo: scikit-learn, xgboost, tensorflow (Keras)
    - Ottimizzazione Prescrittiva: scipy.optimize
===============================================================================
"""

# region LIBRERIE E DICHIARAZIONI 

#===============================================================================
#                           Import librerie
#===============================================================================
import os
# Filtro dei log informativi di TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from scipy.optimize import minimize, OptimizeWarning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


#===============================================================================
#                           Configurazione parametri librerie 
#===============================================================================

# Filtro dei log non critici
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=OptimizeWarning)
tf.get_logger().setLevel('ERROR')

#===============================================================================
#                           Costanti e variabili
#===============================================================================

# Costanti
TEST_SIZE = 0.2         # Percentuale del 20% di test
VALIDATION_SIZE = 0.2   # Percentuale del 20% di parte validazione
RANDOM_SEED = 42       # Inizializzazione del random seed

# Variabili
export_folder = "export_dati"

# Parametri di debug 
SKIP_STEP5 = False
SKIP_STEP6 = False
SKIP_STEP7 = False
PRINT_GRAPH = False  # Visualizzo il grafico dell'errore

# endregion


# Creazione di un nuovo file con aggiunta dei campi di temperatura, velocità,  pressione e difetto
# region STEP 1 

# Init del Random Seed
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Creazione cartella dei file di debug
os.makedirs(export_folder, exist_ok=True)

#===============================================================================
#                           Creazione file con dati di base
#===============================================================================

TARGET_ROWS = 60000

# Init DataFrame vuoto
df = pd.DataFrame(index=range(TARGET_ROWS))

# Definizione delle classi di difetto 
# e relative probabilità di occorrenza 
# Queste percentuali derivano da dati rilevati da varie fonti 
possible_values = [0, 1, 2, 3, 4, 5, 6, 7]
probabilities = [0.980, 0.00165, 0.00195, 0.00405, 0.00075, 0.00055, 0.00415, 0.0069]


# Generazione dei parametri di base  (Temperatura, Velocità, Pressione)
# con distribuzioni normali (Gausssiane)
df['Rolling_Temp_C'] = np.random.normal(loc=950, scale=20, size=TARGET_ROWS)
df['Roller_Speed_m_sec'] = np.random.normal(loc=10, scale=1, size=TARGET_ROWS)
df['Pressure_Bar'] = np.random.normal(loc=200, scale=10, size=TARGET_ROWS)

# Assegnazione casua di difetto in base alle probabilità definite
df['Defects'] = np.random.choice(possible_values, size=TARGET_ROWS, p=probabilities)

#===============================================================================
#              Modifica dei parametri in funzione del difetto
#===============================================================================
# Legenda: 'No_Defects': 0, 'Pastry': 1, 'Z_Scratch': 2, 'K_Scratch': 3, 
# 'Stains': 4, 'Dirtiness': 5, 'Bumps': 6, 'Other_Faults': 7

#1 Pastry (Sfogliatura): Incremento della temperatura
mask_defect = (df['Defects'] == 1) 
df.loc[mask_defect, 'Rolling_Temp_C'] += np.random.uniform(50, 100, size=mask_defect.sum())

#2 Z_Scratch (Graffi Z): Incremento della velocità di laminazione
mask_defect = (df['Defects'] == 2) 
df.loc[mask_defect, 'Roller_Speed_m_sec'] += np.random.uniform(3, 6, size=mask_defect.sum())

#3 K_Scratch (Graffi K): Decremento della velocità di laminazione
mask_defect =  (df['Defects'] == 3)
df.loc[mask_defect, 'Roller_Speed_m_sec'] -= np.random.uniform(3, 6, size=mask_defect.sum())

#4 Stains (Macchie): Decremento della temperatura di laminazione
mask_defect =  (df['Defects'] == 4)
df.loc[mask_defect, 'Rolling_Temp_C'] -= np.random.uniform(50, 100, size=mask_defect.sum())

# 5 Bumps (Irregolarità): Incremento della pressione dei rulli
mask_defect = (df['Defects'] == 5) 
df.loc[mask_defect, 'Pressure_Bar'] += np.random.uniform(40, 80, size=mask_defect.sum())

# 6 Dirtiness (Sporco): Decremento della pressione dei rulli
mask_defect = (df['Defects'] == 6)
df.loc[mask_defect, 'Pressure_Bar'] -= np.random.uniform(40, 80, size=mask_defect.sum())

# 7 Other_Faults (Difetti generici): Modifica combinata,  diminuzione di temperatura e velocità
mask_defect =  (df['Defects'] == 7)
df.loc[mask_defect, 'Rolling_Temp_C'] -= np.random.uniform(50, 100, size=mask_defect.sum())
df.loc[mask_defect, 'Roller_Speed_m_sec'] -= np.random.uniform(1, 3, size=mask_defect.sum())

# Esportazione dati per diagnostica
df.to_excel(f"{export_folder}/01_Step1_Dati_Generati.xlsx", index=False)

# Stampa informazioni di log
print("\n --- STEP 1  10000 righe generate. Variabili simulate corrette.")
averages_df = df.groupby('Defects')[['Rolling_Temp_C', 'Roller_Speed_m_sec', 'Pressure_Bar']].mean().round(2)
print(averages_df)
# endregion


# region STEP 2: Preparazione Target e Features
print("\n--- STEP 2: Preparazione Target e Features ---")

#===============================================================================
#              Separazione Variabili e Mappatura Classi
#===============================================================================

# Separazione delle feature (variabili indipendenti X) dal target (variabile dipendente y)
X = df.drop(columns=['Defects'])

# Definizione del dizionario per la conversione dei codici numerici in testi dei difetti 

defect_map = {
    0: 'No_Defects',
    1: 'Pastry',
    2: 'Z_Scratch',
    3: 'K_Scratch',
    4: 'Stains',
    5: 'Dirtiness',
    6: 'Bumps',
    7: 'Other_Faults'
}

# Conversione della colonna target in formato testuale usando il dizionario
y_names = df['Defects'].map(defect_map)

# Inizializzazione ed esecuzione del LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_names)

# Salvataggio file per diagnostica 
df_step2 = X.copy()
df_step2['Difetto_Target_Numerico'] = y_encoded
df_step2.to_excel(f"{export_folder}/02_Step2_Dati_Ingegnerizzati.xlsx", index=False)

print(f"Dataset pronto per lo split. Feature: {X.shape[1]}, Campioni: {X.shape[0]}")
# endregion


# Split e scaling
#region STEP 3 
print("\n--- STEP 3: Split e Scaling ---")
X_temp, X_test_initial, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded)

# Primo split: isolamento del Test Set
X_train_initial, X_val_initial, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SIZE / (1-TEST_SIZE), random_state=RANDOM_SEED, stratify=y_temp)

# Secondo split: derivazione del Validation Set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_initial)
X_val_scaled = scaler.transform(X_val_initial)
X_test_scaled = scaler.transform(X_test_initial)

# L'operazione di 'fit' viene eseguita solo sul Train Set per prevenire 
# fenomeni di data leakage. Validation e Test Set vengono solo trasformati.
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val = pd.DataFrame(X_val_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Salvataggio per diagnostica
X_train.to_excel(f"{export_folder}/03_Step3_Train_Scalato.xlsx", index=False)
X_val.to_excel(f"{export_folder}/03_Step3_Val_Scalato.xlsx", index=False)
X_test.to_excel(f"{export_folder}/03_Step3_Test_Scalato.xlsx", index=False)

print("\n--- STEP 3: Dati divisi e scalati: ")
print(f"Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
#endregion


#region STEP 4A  XGBoost
from sklearn.model_selection import PredefinedSplit

print("\n--- STEP 4A: Ricerca Iperparametri XGBoost ---")

#===============================================================================
#                               Hyperparameter Tuning
#===============================================================================


# Inizializzazione dell'algoritmo XGBoost
xgb_base = XGBClassifier(
    objective='multi:softprob', 
    random_state=RANDOM_SEED, 
    eval_metric='mlogloss',
    early_stopping_rounds=10   
)

# Definizione dello spazio degli iperparametri
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1]
}

#Concatenazione temporanea di Train e Validation set per la compatibilità con GridSearchCV
X_train_test = pd.concat([X_train, X_val], axis=0) 
y_train_test = np.concatenate([y_train, y_val])    

# Costruzione di un vettore indice per forzare la GridSearchCV a validare 
# esclusivamente sul Validation Set (indice 0), ignorando il Train Set (indice -1)

test_fold = np.concatenate([
    np.full(X_train.shape[0], -1),  
    np.full(X_test.shape[0], 0)     
])
ps = PredefinedSplit(test_fold)

# Configurazione modulo di ottimizzazione
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

# Avvio dell'ottimizzazione
grid_search.fit(
    X_train_test, y_train_test,
    eval_set=[(X_val, y_val)],  
    verbose=False
)

# Estrazione dei risultati della Grid Search in struttura tabellare
results_df = pd.DataFrame(grid_search.cv_results_)

useful_columns = ['param_learning_rate', 'param_max_depth', 'param_n_estimators', 'mean_test_score', 'std_test_score', 'rank_test_score']
results_table = results_df[useful_columns].sort_values(by='rank_test_score')

results_table.rename(columns={
    'param_learning_rate': 'Learning Rate',
    'param_max_depth': 'Max Depth',
    'param_n_estimators': 'N. Alberi',
    'mean_test_score': 'Accuratezza Media',
    'std_test_score': 'Deviazione Standard',
    'rank_test_score': 'Posizione'
}, inplace=True)

results_table['Accuratezza Media'] = results_table['Accuratezza Media'].apply(lambda x: f"{x:.2%}")
results_table['Deviazione Standard'] = results_table['Deviazione Standard'].apply(lambda x: f"{x:.4f}")

print("\n--- I MIGLIORI 5 RISULTATI XGBOOST ---")
print(results_table.head(5).to_string(index=False))

#Salvataggio per diagnostica
results_table.to_excel(f"{export_folder}/04_Step4_Risultati_GridSearch_XGBoost.xlsx", index=False)

# Isolamento della migliore configurazione
model = grid_search.best_estimator_
print(f"\nMigliori iperparametri trovati: {grid_search.best_params_}")

#Isolamento della migliore configurazione
best_parameters = grid_search.best_params_

print("\n Training del modello finale con i parametri migliori")
model = XGBClassifier(
    n_estimators=best_parameters['n_estimators'], 
    learning_rate=best_parameters['learning_rate'], 
    max_depth=best_parameters['max_depth'],
    objective='multi:softprob', 
    random_state=RANDOM_SEED, 
    eval_metric='mlogloss',
    early_stopping_rounds=10  
)

# Train del modello con i migliori parametri e con Early stopping sul dataset di validazione
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=False
)

# Ricerca miglior risultato e numero di iteraizoni
best_iteration = model.best_iteration + 1 
best_score = model.best_score
print(f"Training completato. Il modello si è fermato a {best_iteration} alberi.")
print(f"Miglior errore (mlogloss) su Set Esterno: {best_score:.4f}")

#Stampa del grafico opzionale per vedere come si comporta Early stopping
if PRINT_GRAPH:
    results = model.evals_result()
    train_error = results['validation_0']['mlogloss']
    val_error = results['validation_1']['mlogloss']
    epochs = range(0, len(train_error))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_error, label='Errore di Addestramento (Train)', color='blue')
    plt.plot(epochs, val_error, label='Errore Test Esterno', color='orange') 
    plt.axvline(x=best_iteration, color='red', linestyle='--', label=f'Miglior Iterazione ({best_iteration})')
    plt.title('Curva di Apprendimento XGBoost - Rilevamento Difetti Acciaio')
    plt.xlabel('Numero di Alberi (Iterazioni)')
    plt.ylabel('Errore (Multi-LogLoss)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
#endregion


#region STEP 4B: Rete Neurale (Multi-Layer Perceptron (MLP))
print("\n--- STEP 4B: Ricerca Iperparametri Rete Neurale (TensorFlow) ---")                 

# Definizione dello spazio degli iperparametri
hidden_layer_sizes_options = [(32,), (64,), (32, 16), (64, 32)]    # 4 architetture (da semplici a profonde a imbuto)
activation_options = ['relu', 'tanh']                             # 2 funzioni di attivazione classiche
learning_rate_options = [0.005, 0.01]                             # 2 velocità (Adam di default a 0.001, e una più aggressiva a 0.01)
alpha_options = [0.0001, 0.01]

# Isolamento delle variabili di validazione
X_eval_keras = X_val
y_eval_keras = y_val

#===============================================================================
#              Factory Pattern per Generazione Dinamica del Modello
#===============================================================================

# Funzione per costruire il modello Keras dinamicamente
def build_keras_model(hidden_layers, activation, lr, alpha, input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    # Aggiunta dinamica dei layer nascosti
    model.add(Dense(hidden_layers[0], activation=activation, kernel_regularizer=l2(alpha)))
    if len(hidden_layers) > 1:
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation, kernel_regularizer=l2(alpha)))
    # Layer di output multiclasse (8 nodi) con attivazione Softmax per conversione in probabilità
    model.add(Dense(8, activation='softmax'))

    #Compilazione del grafo computazionale
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Generazione del prodotto cartesiano di tutti gli iperparametri definiti
combinations = list(itertools.product(hidden_layer_sizes_options, activation_options, learning_rate_options, alpha_options))
print(f"Inizio addestramento per {len(combinations)} combinazioni (Keras). Attendere...")
results_keras = []
best_keras_model = None
best_loss = float('inf')
best_accuracy = 0.0
best_parameters_keras = {}
best_history = None

# Configurazione del meccanismo di interruzione anticipata (Early Stopping)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True,
    verbose=0
)

# Ciclo di validazione delle combinazioni
for i, (hl, act, lr, alpha) in enumerate(combinations):
    print(f"[{i+1}/{len(combinations)}] Addestramento rete {hl} | {act} | LR:{lr}...")
    
    temp_model = build_keras_model(hl, act, lr, alpha, input_dim=X_train.shape[1])
    
    history = temp_model.fit(
        X_train, y_train,
        validation_data=(X_eval_keras, y_eval_keras),
        epochs=200,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=0  
    )
    
    # Valutazione della combinazione
    val_loss, val_acc = temp_model.evaluate(X_eval_keras, y_eval_keras, verbose=0)
    completed_iterations = len(history.history['loss'])
    
    results_keras.append({
        'Neuroni e Livelli Nascosti': str(hl),
        'Attivazione': act,
        'Learning Rate': lr,
        'Regolarizzazione (Alpha)': alpha,
        'Accuratezza Media': val_acc,
        'Loss Esterna': val_loss,
        'Iterazioni Effettive': completed_iterations
    })
    
    # Se l'accuratezza è maggiore, o se c'è un pareggio ma la loss è minore:
    if val_acc > best_accuracy or (val_acc == best_accuracy and val_loss < best_loss):
        best_accuracy = val_acc
        best_loss = val_loss
        best_keras_model = temp_model
        best_history = history
        best_parameters_keras = {'hidden_layers': hl, 'activation': act, 'lr': lr, 'alpha': alpha}

# Creazione della tabella dei results
df_results_keras = pd.DataFrame(results_keras).sort_values(by=['Accuratezza Media', 'Loss Esterna'], ascending=[False, True])
df_results_keras['Accuratezza Media'] = df_results_keras['Accuratezza Media'].apply(lambda x: f"{x:.2%}")

print("\n--- I MIGLIORI RISULTATI RETE NEURALE (TENSORFLOW) ---")
print(df_results_keras.head(5).to_string(index=False))

df_results_keras.to_excel(f"{export_folder}/04_Step4B_Risultati_TensorFlow.xlsx", index=False)


#Stampa del grafico opzionale per vedere come si comporta Early stopping
if PRINT_GRAPH:
    plt.figure(figsize=(10, 6))
    train_error_keras = best_history.history['loss']
    val_error_keras = best_history.history['val_loss']
    epochs_keras = range(1, len(train_error_keras) + 1)
    
    plt.plot(epochs_keras, train_error_keras, label='Curva di Errore (Train)', color='green', linewidth=2)
    plt.plot(epochs_keras, val_error_keras, label='Curva di Errore (Set Esterno)', color='purple', linewidth=2)
    
    plt.title('Curva di Apprendimento Rete Neurale - Rilevamento Difetti')
    plt.xlabel('Iterazioni (Epoche)')
    plt.ylabel('Errore (Loss)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

#endregion

# Valutazione comparativa
# region STEP 5

# Salto fase per debug sezioni precedenti
if not SKIP_STEP5:
    print("\n=== STEP 5: VALUTAZIONE COMPARATIVA ===")
    target_names = label_encoder.classes_

    # Valutazione XGBoost
    y_pred_xgb = model.predict(X_test)
    print("\n--- METRICHE XGBOOST ---")
    print(f"Accuratezza Globale: {accuracy_score(y_test, y_pred_xgb):.2%}")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names, zero_division=0))

    # Stampa la matrice di confusione XGBoost
    print("\n--- MATRICE DI CONFUSIONE XGBOOST ---")
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    df_cm_xgb = pd.DataFrame(cm_xgb, index=target_names, columns=target_names)
    print(df_cm_xgb)

    # Valutazione MLP (Rete Neurale) 
    mlp_probabilities = best_keras_model(np.array(X_test), training=False).numpy()
    y_pred_mlp = np.argmax(mlp_probabilities, axis=1)
    print("\n--- METRICHE RETE NEURALE (TENSORFLOW) ---")
    print(f"Accuratezza Globale: {accuracy_score(y_test, y_pred_mlp):.2%}")
    print(classification_report(y_test, y_pred_mlp, target_names=target_names, zero_division=0))

    # Stampa la matrice di confusione MLP
    print("\n--- MATRICE DI CONFUSIONE RETE NEURALE ---")
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    df_cm_mlp = pd.DataFrame(cm_mlp, index=target_names, columns=target_names)
    print(df_cm_mlp)


#endregion

#region STEP 6: Simulazione Prescrittiva su 100 Barre
if not SKIP_STEP6:
    print("\n--- STEP 6: Simulazione Prescrittiva su 100 Barre ---")

    NUM_TEST = 100
    simulation_results = []

    # Ricerca di tutti i potenziali casi critici (usando XGBoost come "ispettore" base)
    reference_probs = model.predict_proba(X_test)
    idx_no_defect = list(label_encoder.classes_).index('No_Defects')
    
    probs_only_defects = reference_probs.copy()
    probs_only_defects[:, idx_no_defect] = 0.0

    max_probs = np.max(probs_only_defects, axis=1)
    defective_indices = np.where(max_probs > 0.5)[0]

    # Selezie di 100 indici casuali tra quelli difettosi
    test_indices = np.random.choice(defective_indices, size=NUM_TEST, replace=len(defective_indices) < NUM_TEST)

    cols = X_test.columns.tolist()
    idx_temp = cols.index('Rolling_Temp_C')
    idx_speed = cols.index('Roller_Speed_m_sec')
    idx_press = cols.index('Pressure_Bar')

    # Modelli da testare
    models_to_test = [
        ("XGBoost", model),
        ("TensorFlow", best_keras_model)
    ]

    print(f"Avvio ottimizzazione per {NUM_TEST} barre su entrambi i modelli. Attendere...")

    # 2. Eseguo il ciclo sui 100 casi
    for i, case_idx in enumerate(test_indices):
        # Stampa avanzamento
        if (i+1) % 10 == 0:
            print(f"Progresso: {i+1}/{NUM_TEST} barre completate...")

        # Estrazione del campione e scalatura in valori reali 
        scaled_row = X_test.iloc[case_idx].values.reshape(1, -1)
        real_row = scaler.inverse_transform(scaled_row)
        real_case = pd.Series(real_row[0], index=X_test.columns)

        # Difetto da risolvere
        predicted_defect_idx = np.argmax(reference_probs[case_idx])
        defect_name = label_encoder.inverse_transform([predicted_defect_idx])[0]

        #Vettore di partenza limitato
        temp_start = max(800, min(1000, real_case['Rolling_Temp_C']))
        speed_start = max(9, min(15, real_case['Roller_Speed_m_sec']))
        press_start = max(150, min(300, real_case['Pressure_Bar']))
        x0 = [temp_start, speed_start, press_start]
        bounds = [(800, 1000), (9, 15), (150, 300)]

        bar_result = {
            'Id_Barra': i+1,
            'Difetto': defect_name,
            'Temp_Iniziale': real_case['Rolling_Temp_C'],
            'Speed_Iniziale': real_case['Roller_Speed_m_sec'],
            'Press_Iniziale': real_case['Pressure_Bar']
        }

        # Testo entrambi i modelli sulla stessa barra
        for model_name, current_model in models_to_test:
            
          # calcolo della probabilità di errore
            def get_probability(model, name, data):
                if name == "XGBoost":
                    return model.predict_proba(data)[0][predicted_defect_idx]
                else: # TensorFlow
                    return model(np.array(data), training=False).numpy()[0][predicted_defect_idx]
            
            initial_prob = get_probability(current_model, model_name, scaled_row)

            # Funzione obiettivo con penalità
            def objective_function(x_new):
                row_simulation = real_case.values.copy() 
                row_simulation[idx_temp] = x_new[0]         
                row_simulation[idx_speed] = x_new[1]        
                row_simulation[idx_press] = x_new[2]        
            
                row_scaled = scaler.transform(row_simulation.reshape(1, -1))
                
                # Uso lo smistatore invece del wrapper
                defect_prob = get_probability(current_model, model_name, row_scaled)

                temp_penalty = ((x_new[0] - real_case['Rolling_Temp_C']) / real_case['Rolling_Temp_C'])**2
                speed_penalty = ((x_new[1] - real_case['Roller_Speed_m_sec']) / real_case['Roller_Speed_m_sec'])**2
                press_penalty = ((x_new[2] - real_case['Pressure_Bar']) / real_case['Pressure_Bar'])**2
                
                return defect_prob + (0.5 * (temp_penalty + speed_penalty + press_penalty))

        
            # Inizio misura tempo di esecuzione
            optimization_start = time.perf_counter()

            # Minimizzo la funzione
            result = minimize(objective_function, x0, method='Powell', bounds=bounds, tol=1e-3, options={'maxiter': 30})

            # Fine misura tempo di esecuzione
            optimization_end = time.perf_counter()
            elapsed_time = optimization_end - optimization_start
            

            bar_result[f'Probabilità_Iniziale_{model_name}'] = initial_prob
            bar_result[f'Tempo_Esecuzione_{model_name}'] = elapsed_time 
            
            if result.success:
                final_row = real_case.values.copy()
                final_row[idx_temp], final_row[idx_speed], final_row[idx_press] = result.x
                row_df_fin = pd.DataFrame(final_row.reshape(1, -1), columns=X_test.columns)
                new_prob = get_probability(current_model, model_name, scaler.transform(row_df_fin))
                
                bar_result[f'Prob_Finale_{model_name}'] = new_prob
                bar_result[f'Temp_Finale_{model_name}'] = result.x[0]
                bar_result[f'Press_Finale_{model_name}'] = result.x[2]
                bar_result[f'Vel_Finale_{model_name}'] = result.x[1]
                bar_result[f'Delta_Temp_{model_name}'] = abs(result.x[0] - real_case['Rolling_Temp_C'])
                bar_result[f'Delta_Press_{model_name}'] = abs(result.x[2] - real_case['Pressure_Bar'])
                bar_result[f'Delta_Vel_{model_name}'] = abs(result.x[1] - real_case['Roller_Speed_m_sec'])
                bar_result[f'Successo_{model_name}'] = True
            else:
                bar_result[f'Prob_Finale_{model_name}'] = initial_prob
                bar_result[f'Temp_Finale_{model_name}'] = 0
                bar_result[f'Press_Finale_{model_name}'] = 0
                bar_result[f'Vel_Finale_{model_name}'] = 0
                bar_result[f'Delta_Temp_{model_name}'] = 0
                bar_result[f'Delta_Press_{model_name}'] = 0
                bar_result[f'Delta_Vel_{model_name}'] = 0
                bar_result[f'Successo_{model_name}'] = False

        # Aggiungo la barra completata alla lista dei results
        simulation_results.append(bar_result)

    # Salvo in excel per diagnostica
    simulation_df = pd.DataFrame(simulation_results)
    simulation_df.to_excel(f"{export_folder}/06_Step6_Simulazione_Massiva_100.xlsx", index=False)
    print(f"\nSimulazione completata. Dati grezzi salvati in '{export_folder}/06_Step6_Simulazione_Massiva_100.xlsx'")

#endregion

#region STEP 7: Report Statistico sull'analisi prescrittiva
if not SKIP_STEP7:
    print("\n=== STEP 7: REPORT STATISTICO SULL'ANALISI PRESCRITTIVA ===")

    # Calcolo della probabilità media residua di anomalia post-ottimizzazione
    avg_final_prob_xgb = simulation_df['Prob_Finale_XGBoost'].mean()
    avg_final_prob_mlp = simulation_df['Prob_Finale_TensorFlow'].mean()

    # Quantificazione dello scostamento medio assoluto (Delta) imposto ai parametri
    Delta_Temp_XGBoost = simulation_df['Delta_Temp_XGBoost'].mean()
    Delta_Temp_TensorFlow = simulation_df['Delta_Temp_TensorFlow'].mean()

    Delta_Press_XGBoost = simulation_df['Delta_Press_XGBoost'].mean()
    Delta_Press_TensorFlow = simulation_df['Delta_Press_TensorFlow'].mean()

    Delta_Spd_XGBoost = simulation_df['Delta_Vel_XGBoost'].mean()
    Delta_Spd_TensorFlow = simulation_df['Delta_Vel_TensorFlow'].mean()

    # Calcolo del tempo medio di prescrizione per barra
    avg_time_xgb = simulation_df['Tempo_Esecuzione_XGBoost'].mean()
    avg_time_mlp= simulation_df['Tempo_Esecuzione_TensorFlow'].mean()
    total_time_xgb = simulation_df['Tempo_Esecuzione_XGBoost'].sum()
    total_time_mlp = simulation_df['Tempo_Esecuzione_TensorFlow'].sum()

    # Calcolo di quante volte il modello è riuscito a portare il rischio sotto una soglia di sicurezza (< 5%)
    safe_bars_xgb = (simulation_df['Prob_Finale_XGBoost'] < 0.05).sum()
    safe_bars_mlp = (simulation_df['Prob_Finale_TensorFlow'] < 0.05).sum()

    # Stampa del verdetto finale
    print("\n1. CAPACITÀ DI RISOLUZIONE DEL DIFETTO:")
    print(f" - Barre curate in sicurezza (<5% rischio) da XGBoost:    {safe_bars_xgb}/{NUM_TEST}")
    print(f" - Barre curate in sicurezza (<5% rischio) da TensorFlow: {safe_bars_mlp}/{NUM_TEST}")
    print(f" - Rischio medio residuo dopo prescrizione XGBoost:       {avg_final_prob_xgb:.2%}")
    print(f" - Rischio medio residuo dopo prescrizione TensorFlow:    {avg_final_prob_mlp:.2%}")

    print("\n2. VARIAZIONI MEDIE RICHIESTE ALL'IMPIANTO:")
    print(f" - Variazione Temperatura XGBoost:    {Delta_Temp_XGBoost:.2f} °C")
    print(f" - Variazione Temperatura TensorFlow: {Delta_Temp_TensorFlow:.2f} °C")
    print(f" - Variazione Pressione XGBoost:      {Delta_Press_XGBoost:.2f} bar")
    print(f" - Variazione Pressione TensorFlow:   {Delta_Press_TensorFlow:.2f} bar")
    print(f" - Variazione Velocità XGBoost:       {Delta_Spd_XGBoost:.2f} m/s")
    print(f" - Variazione Velocità TensorFlow:    {Delta_Spd_TensorFlow:.2f} m/s")

    print("\n3. PRESTAZIONI COMPUTAZIONALI (TEMPI DI OTTIMIZZAZIONE):")
    print(f" - Tempo medio per barra XGBoost:     {avg_time_xgb:.4f} secondi")
    print(f" - Tempo medio per barra TensorFlow:  {avg_time_mlp:.4f} secondi")
    print(f" - Tempo totale per 100 barre XGBoost:{total_time_xgb:.2f} secondi")
    print(f" - Tempo totale per 100 barre TF:     {total_time_mlp:.2f} secondi")

    # Selezione del modello vincitore
    if (avg_final_prob_xgb < avg_final_prob_mlp):
        winner = "XGBoost"
    else:
        winner = "Rete Neurale (TensorFlow)"   
    print(f"\n VERDETTO DELLA SIMULAZIONE: {winner} è il modello più efficace nella prescrizione!")

    # Salvataggio dei valori per diagnostica
    with open(f"{export_folder}/07_Step7_Report_Statistico.txt", "w") as file:
        file.write("=== REPORT STATISTICO SULL'ANALISI PRESCRITTIVA (100 BARRE) ===\n\n")
        file.write(f"Barre curate in sicurezza (<5%) da XGBoost:    {safe_bars_xgb}/{NUM_TEST}\n")
        file.write(f"Barre curate in sicurezza (<5%) da TensorFlow: {safe_bars_mlp}/{NUM_TEST}\n")
        file.write(f"Variazione media Temperatura XGBoost:          {Delta_Temp_XGBoost:.2f} °C\n")
        file.write(f"Variazione media Temperatura TensorFlow:       {Delta_Temp_TensorFlow:.2f} °C\n")
        file.write(f"Variazione media Pressione XGBoost:            {Delta_Press_XGBoost:.2f} bar\n")
        file.write(f"Variazione media Pressione TensorFlow:         {Delta_Press_TensorFlow:.2f} bar\n")
        file.write(f"Variazione media Velocità XGBoost:             {Delta_Spd_XGBoost:.2f} m/s\n")
        file.write(f"Variazione media Velocità TensorFlow:          {Delta_Spd_TensorFlow:.2f} m/s\n")
        file.write(f"Vincitore Globale:                             {winner}\n")
        file.write(f"\nTempo medio per barra XGBoost:                 {avg_time_xgb:.4f} sec\n")
        file.write(f"Tempo medio per barra TensorFlow:              {avg_time_mlp:.4f} sec\n")
        file.write(f"Tempo totale (100 barre) XGBoost:              {total_time_xgb:.2f} sec\n")
        file.write(f"Tempo totale (100 barre) TensorFlow:           {total_time_mlp:.2f} sec\n")
#endregion