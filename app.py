import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Importaciones de TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuración de página
st.set_page_config(page_title="Predicción Deserción Universitaria", layout="wide")

# Título principal de la aplicación
st.title("Sistema de Predicción de Deserción Universitaria")
st.markdown("### Enfoque Actualizado: Red Neuronal con Regularización Dropout")

# --- CARGA Y PREPROCESAMIENTO DE DATOS EN CACHÉ ---
@st.cache_data
def load_and_clean_data():
    try:
        data = pd.read_csv('data.csv', sep=';')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'data.csv'. Por favor, colócalo en el mismo directorio.")
        st.stop()
    
    # Limpieza de nombres de columnas
    data.columns = data.columns.str.strip()
    
    # Excluir a los estudiantes 'Enrolled'
    data = data[data['Target'] != 'Enrolled'].copy()
    
    # Mapa de conversión binario limpio (Graduate = 1, Dropout = 0)
    data['Target_Binario'] = data['Target'].map({'Graduate': 1, 'Dropout': 0}).astype(int)
    return data

data = load_and_clean_data()

# Separar características (X) y etiqueta (y)
X = data.drop(columns=['Target', 'Target_Binario'])
y = data['Target_Binario']

# Dividir en entrenamiento y prueba de manera fija (Idéntico al script)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
@st.cache_resource
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# --- CARGA O ENTRENAMIENTO DEL MODELO NEURONAL ---
@st.cache_resource
def get_neural_network_model(_X_train_scaled, _y_train):
    model_path = 'modelo_dropout.keras'
    history_path = 'historial_modelo.json'
    
    class LoadedHistory:
        def __init__(self, history_dict):
            self.history = history_dict

    # Forzar la carga si los archivos del script ya existen en la carpeta
    if os.path.exists(model_path) and os.path.exists(history_path):
        model = load_model(model_path)
        with open(history_path, 'r') as f:
            history_dict = json.load(f)
        history = LoadedHistory(history_dict)
    else:
        # Configuración de contingencia si no se encuentra el archivo pre-entrenado
        num_caracteristicas = _X_train_scaled.shape[1]
        model = Sequential([
            Dense(64, activation='relu', input_shape=(num_caracteristicas,), name="Capa_Entrada_Oculta1"),
            Dropout(0.3, name="Dropout_1"),
            Dense(32, activation='relu', name="Capa_Oculta2"),
            Dropout(0.2, name="Dropout_2"),
            Dense(16, activation='relu', name="Capa_Oculta3"),
            Dense(1, activation='sigmoid', name="Capa_Salida")
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        fit_history = model.fit(
            _X_train_scaled, _y_train,
            epochs=60, batch_size=32,
            validation_split=0.2, callbacks=[early_stop], verbose=0
        )
        
        model.save(model_path)
        with open(history_path, 'w') as f:
            json.dump(fit_history.history, f)
            
        history = LoadedHistory(fit_history.history)
        
    return model, history

model_dropout, history = get_neural_network_model(X_train_scaled, y_train)

# --- BARRA LATERAL (NAVEGACIÓN) ---
st.sidebar.header("Menú de Navegación")
menu = st.sidebar.selectbox(
    "Selecciona la funcionalidad", 
    ["Exploración (EDA)", "Predicción Interactiva Neuronal", "Evaluación y Métricas"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**EQUIPO DE TRABAJO:**")
st.sidebar.markdown(
    "1. Hinsna Jurany Rincon Guerrero\n"
    "2. Ruth Catherin Rincon Guerrero\n"
    "3. Deivy Steve Varon Corredor\n"
    "4. Julio Andrés Silva Aragón\n"
    "5. Rigo Arvey Chapid Guacales"
)

# --- PANEL 1: EXPLORACIÓN (EDA) ---
if menu == "Exploración (EDA)":
    st.header("1. Análisis Exploratorio de Datos (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de la Edad al Matricularse")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=data, x='Age at enrollment', hue='Target', multiple='stack', bins=30, palette='viridis', ax=ax)
        ax.set_xlabel('Edad al Matricularse (Age at enrollment)')
        ax.set_ylabel('Cantidad de Estudiantes')
        st.pyplot(fig)
        st.caption("Conclusión: El riesgo de abandono escolar se incrementa exponencialmente a mayor edad de inscripción.")
    with col2:
        st.subheader("Calificaciones del 2do Semestre")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=data, x='Target', y='Curricular units 2nd sem (grade)', palette='Pastel1', ax=ax)
        ax.set_xlabel('Estado Final')
        ax.set_ylabel('Nota Promedio (2do Semestre)')
        st.pyplot(fig)
        st.caption("Conclusión: Los estudiantes que abandonan suelen acumular notas promedio cercanas a cero.")

# --- PANEL 2: PREDICCIÓN INTERACTIVA NEURONAL ---
elif menu == "Predicción Interactiva Neuronal":
    st.header("2. Módulo de Predicción con Red Neuronal Profunda")
    st.write("La evaluación se procesa mediante una arquitectura secuencial con capas densas y regularización Dropout.")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Edad al matricularse (Age at enrollment):", min_value=10, max_value=70, value=20)
        gender = st.selectbox("Género (Gender):", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        scholarship = st.selectbox("¿Tiene beca? (Scholarship holder):", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    with col2:
        tuition = st.selectbox("¿Matrícula al día? (Tuition fees up to date):", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        debtor = st.selectbox("¿Es deudor? (Debtor):", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        admission_grade = st.slider("Nota de admisión (Admission grade):", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
    with col3:
        enrolled_2nd = st.number_input("Unidades curriculares inscritas 2° sem:", min_value=0, max_value=30, value=6)
        approved_2nd = st.number_input("Unidades curriculares aprobadas 2° sem:", min_value=0, max_value=30, value=5)
        grade_2nd = st.slider("Nota promedio 2° semestre:", min_value=0.0, max_value=20.0, value=12.0, step=0.1)

    if st.button("Procesar por Red Neuronal"):
        input_data = X.mean().to_frame().T
        input_data['Age at enrollment'] = age
        input_data['Gender'] = gender
        input_data['Scholarship holder'] = scholarship
        input_data['Tuition fees up to date'] = tuition
        input_data['Debtor'] = debtor
        input_data['Admission grade'] = admission_grade
        input_data['Curricular units 2nd sem (enrolled)'] = enrolled_2nd
        input_data['Curricular units 2nd sem (approved)'] = approved_2nd
        input_data['Curricular units 2nd sem (grade)'] = grade_2nd
        
        input_scaled = scaler.transform(input_data)
        prob = model_dropout.predict(input_scaled)[0][0]
        pred_class = 1 if prob > 0.5 else 0
        st.markdown("---")
        if pred_class == 1:
            st.success(f"### Estado Predicho por Red Neuronal: **GRADUADO (Graduate)**")
            st.metric(label="Probabilidad de éxito académico (Salida Sigmoide)", value=f"{prob*100:.2f}%")
        else:
            st.error(f"### Estado Predicho por Red Neuronal: **DESERCIÓN (Dropout)**")
            st.metric(label="Probabilidad de deserción estimada", value=f"{(1-prob)*100:.2f}%")

# --- PANEL 3: EVALUACIÓN Y MÉTRICAS ---
elif menu == "Evaluación y Métricas":
    st.header("3. Curvas de Aprendizaje y Robustez del Modelo Neuronal")
    st.write("Análisis detallado de la convergencia (Loss y Accuracy) y distribución del error.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'], label='Entrenamiento', color='tab:blue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validación', color='tab:orange', linewidth=2)
    ax1.set_title('Función de Pérdida (Binary Crossentropy)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Entrenamiento', color='tab:green', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validación', color='tab:red', linewidth=2)
    ax2.set_title('Exactitud del Modelo (Accuracy)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Exactitud')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig)
    
    # Inferencia en el Set de Test
    y_pred_probs = model_dropout.predict(X_test_scaled)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    
    # SECCIÓN CORREGIDA CON INDENTACIÓN ADECUADA
    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.subheader("Matriz de Confusión en Test")
        cm_neural = confusion_matrix(y_test, y_pred_classes)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_neural, display_labels=['Dropout (0)', 'Graduate (1)'])
        disp.plot(cmap=plt.cm.Purples, ax=ax_cm, values_format='d')
        st.pyplot(fig_cm)
    with col_metrics2:
        st.subheader("Informe de Clasificación Robustas")
        report_dict = classification_report(y_test, y_pred_classes, target_names=['Dropout (0)', 'Graduate (1)'], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        accuracy_global = report_dict['accuracy']
        st.metric(label="Exactitud Global de la Red Neuronal (Test Accuracy)", value=f"{accuracy_global*100:.2f}%")
