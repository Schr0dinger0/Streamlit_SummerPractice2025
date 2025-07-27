import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier, Pool
import numpy as np
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots




# Настройки страницы
st.set_page_config(layout="wide")
st.title("Корязов Д. И._2023-ФГиИБ-ПИ-1б_Вариант_13_Bank_Marketing")

# Загрузка данных и модели
def load_data(name):
    return pd.read_csv(name)  # Ваш датасет

def load_model():
    return CatBoostClassifier().load_model("catboost_model.cbm")

data = load_data("final_data.csv")
model = load_model()

data_shap = data.copy()
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
    data_shap[col] = data_shap[col].map({'no': 0, 'yes': 1})



selected = st.sidebar.radio("Навигация", ["Model metrics", "Dataset info", "HeatMap"])
if selected == "Dataset info":
    # Описание датасета
    st.header("Описание набора данных")
    st.markdown("""
    Данные содержат информацию о маркетинговой кампании банка:
    - **16 признаков**: возраст, профессия, семейное положение и др.
    - **Целевая переменная**: подписал ли клиент депозит (yes/no)
    - **Размер**: 4521 записей
    """)
    
    # Загрузка данных
    @st.cache_data
    def load_data():
        return pd.read_csv("final_data.csv")
    
    data = load_data()
    
    # Выбор категориальных переменных
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox(
            "Выберите первую категориальную переменную:",
            options=categorical_cols,
            index=categorical_cols.index('poutcome') if 'poutcome' in categorical_cols else 0
        )
    
    with col2:
        var2 = st.selectbox(
            "Выберите вторую категориальную переменную:",
            options=categorical_cols,
            index=categorical_cols.index('y') if 'y' in categorical_cols else 1
        )
    
    # Создаем кросс-таблицу с долями
    contingency_table = pd.crosstab(data[var1], data[var2])
    contingency_table_norm = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    # Создаем тепловую карту
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(
        contingency_table_norm,
        annot=True,
        fmt=".1f",
        cmap='Reds',
        linewidths=.3,
        ax=ax
    )
    
    # Настраиваем внешний вид
    plt.title(f'Доли {var2} в зависимости от {var1}, %')
    plt.xlabel(var2)
    plt.ylabel(var1)
    plt.xticks(rotation=45)
    
    # Отображаем график в Streamlit
    st.pyplot(fig, use_container_width=False)
    
    # Дополнительная информация
    st.markdown(f"""
    **Интерпретация:**
    - Таблица показывает распределение (%) значений **{var2}** для каждого значения **{var1}**
    - Например: {contingency_table_norm.iloc[0,0]:.1f}% наблюдений с {var1}={contingency_table_norm.index[0]} 
      имеют {var2}={contingency_table_norm.columns[0]}
    """)
elif selected == "HeatMap":
    # Настройки страницы
    st.set_page_config(layout="wide")
    st.title("Анализ корреляций данных")

    # Загрузка данных
    @st.cache_data
    def load_data(version):
        if version == "Исходные":
            df = pd.read_csv("data.csv")
        else:
            df = pd.read_csv("final_data.csv")
        
        # Преобразование категориальных переменных
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]
        return df

    # Выбор версии данных
    data_version = st.radio(
        "Выберите версию данных для анализа:",
        ("Исходные", "Финальные"),
        horizontal=True
    )

    data = load_data(data_version)
    corr_matrix = data.corr().round(2)

    # Создаем текст для hover-подсказок
    hover_text = []
    for yi in corr_matrix.index:
        row = []
        for xi in corr_matrix.columns:
            val = corr_matrix.loc[yi, xi]
            if yi == xi:
                row.append(f"<b>{yi} ↔ {xi}</b><br>Корреляция = 1.0")
            else:
                row.append(f"<b>{yi} ↔ {xi}</b><br>Корреляция = {val:.2f}")
        hover_text.append(row)

    # Создаем тепловую карту
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        text=hover_text,
        hoverinfo="text",
        colorbar=dict(title='Коэффициент корреляции')
    ))

    # Настраиваем размеры и внешний вид
    num_params = len(corr_matrix.columns)
    cell_size = 60 
    plot_size = max(900, num_params * cell_size)

    fig.update_layout(
        title=f'Матрица корреляций ({data_version} данные)',
        width=plot_size,
        height=plot_size,
        xaxis=dict(
            tickangle=45,
            constrain='domain',
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            constrain='domain'
        ),
        margin=dict(l=100, r=100, t=100, b=100),
        hoverlabel=dict(
            bgcolor="black",
            font_size=12,
            font_family="Arial"
        )
    )

    # Добавляем аннотации
    annotations = []
    threshold = 0.25  # Порог для отображения значений

    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            if abs(corr_matrix.iloc[i, j]) > threshold and i != j:
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=str(corr_matrix.iloc[i, j]),
                        showarrow=False,
                        font=dict(
                            color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                            size=10
                        )
                    )
                )

    fig.update_layout(annotations=annotations)

    # Отображаем график
    st.plotly_chart(fig, use_container_width=True)

    # Добавляем пояснения
    st.markdown("""
    **Интерпретация корреляций:**
    - От +0.7 до +1.0: сильная положительная связь
    - От +0.3 до +0.7: умеренная положительная связь
    - От -0.3 до +0.3: слабая или отсутствует связь
    - От -0.7 до -0.3: умеренная отрицательная связь
    - От -1.0 до -0.7: сильная отрицательная связь

    <small>Значения корреляции по модулю >0.25 отображаются на карте</small>
    """, unsafe_allow_html=True)
elif selected == "Model metrics":
    # Разделение на 2 колонки
    col1, col2 = st.columns([1, 1])

    # Колонка 1: Метрики модели
    with col1:
        st.header("Оценка точности модели")
        met1, met2, met3 = st.columns([1,1,1])
        with met1:
            st.metric("Accuracy", "89.83%")
        with met2:
            st.metric("Recall (класс 'yes')", "79%")
        with met3:
            st.metric("Precision (класс 'yes')", "55%")
        
        # Матрица ошибок
        st.subheader("Confusion Matrix")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
        fig.delaxes(ax2)  # Удаляем второй subplot
        cm = [[727, 69], [23, 86]]
        ax1.matshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax1.text(j, i, val, ha='center', va='center', fontsize = 6)
        plt.xticks([0, 1], ["Предсказано no", "Предсказано yes"], fontsize = 4)
        plt.yticks([0, 1], ["Правда no", "Правда yes"], fontsize = 4)
        st.pyplot(fig)

    # Колонка 2: Анализ признаков
    with col2:
        st.header("Анализ влияния признаков")
        
        cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'poutcome', 'balance_group', 'age_group', 'day_group','pdays_group', 
                    'duration_group', 'campaign_group']

        # SHAP-график
        st.subheader("Важность признаков (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Pool(data_shap.sample(100), cat_features=cat_features))
        fig, ax1 = plt.subplots()
        shap.summary_plot(shap_values, data_shap.sample(100), plot_type="bar", show=False)
        st.pyplot(fig)