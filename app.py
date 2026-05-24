import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, gaussian_kde
from scipy.signal import find_peaks
import re

# Configuração da Página
st.set_page_config(page_title="Análise Estatística & Regressão", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2C3E50; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Plataforma de Análise Estatística e Regressão Linear")
st.markdown("Faça o upload da sua base de dados, defina as variáveis e obtenha uma análise acadêmica completa automatizada.")

# Funções Auxiliares de Análise
def calcular_descritiva(df, cols):
    stats = []
    for c in cols:
        s = df[c].dropna()
        mean = s.mean()
        std = s.std()
        stats.append({
            'Variável': c,
            'Média': mean,
            'Mediana': s.median(),
            'Moda': s.mode().iloc[0] if not s.mode().empty else np.nan,
            'Desv Padrão': std,
            'Variância': s.var(),
            'CV (%)': (std / mean * 100) if mean != 0 else np.nan,
            'Mínimo': s.min(),
            'Máximo': s.max(),
            'Amplitude': s.max() - s.min(),
            'Q1': s.quantile(0.25),
            'Q3': s.quantile(0.75)
        })
    return pd.DataFrame(stats).set_index('Variável')

def analisar_assimetria(s):
    if s.nunique() <= 1: return "Constante (sem variação)"
    sk = skew(s.dropna())
    if sk > 0.5:
        return "Assimétrica Positiva (cauda à direita)"
    elif sk < -0.5:
        return "Assimétrica Negativa (cauda à esquerda)"
    return "Relativamente Simétrica"

def detectar_bimodalidade(s):
    s_clean = s.dropna()
    if len(s_clean) < 10 or s_clean.nunique() <= 1: return False
    try:
        kde = gaussian_kde(s_clean)
        x_range = np.linspace(s_clean.min(), s_clean.max(), 100)
        peaks, _ = find_peaks(kde(x_range))
        return len(peaks) > 1
    except:
        return False

def format_p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

def formatar_texto_latex(texto):
    txt = str(texto).replace('%', '\\%').replace('$', '\\$').replace('_', '\\_')
    return f"\\text{{{txt}}}"

def recuperar_nota_corrompida(val):
    val_str = str(val).strip()
    # Verifica se o Excel corrompeu a nota transformando-a em formato de data YYYY-MM-DD
    match = re.match(r'^2026[-/](\d{2})[-/](\d{2})', val_str)
    if match:
        mes = int(match.group(1))
        dia = int(match.group(2))
        # No Excel corrompido: 4.7 vira 2026-07-04 (Mês = decimal, Dia = inteiro) ou vice-versa.
        # Geralmente o dia é a parte inteira (4) e o mês o decimal (7), gerando 4.7
        if dia <= 5:
            return float(f"{dia}.{mes}")
        else:
            return float(f"{mes}.{dia}")
    
    # Se for formato DD/MM/2026 ou similar
    match_br = re.match(r'^(\d{2})[-/](\d{2})[-/](2026|\d{2})', val_str)
    if match_br:
        d = int(match_br.group(1))
        m = int(match_br.group(2))
        if d <= 5: return float(f"{d}.{m}")
        if m <= 5: return float(f"{m}.{d}")
        
    # Limpeza padrão para números normais contendo vírgulas ou símbolos
    limpo = val_str.replace(',', '.')
    limpo = re.sub(r'[^\d\.\-]+', '', limpo)
    try:
        return float(limpo) if limpo else np.nan
    except:
        return np.nan

# Interface Lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ Configurações do Modelo")
    uploaded_file = st.file_uploader("1. Upload da Base de Dados", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # --- TRATAMENTO CORRETIVO DAS VARIÁVEIS ---
            for col in df.columns:
                if str(col).lower() not in ['obs', 'obs.', 'id', 'identificação', 'unidade', 'região']:
                    # Aplica a recuperação inteligente contra erros de conversão do Excel
                    df[col] = df[col].apply(recuperar_nota_corrompida)

            # Captura colunas estritamente numéricas e úteis
            all_numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if str(c).lower() not in ['obs', 'obs.', 'id']]
            all_numeric_cols = [c for c in all_numeric_cols if df[c].notna().sum() > 0]

            if len(all_numeric_cols) < 2:
                st.error("A base precisa ter pelo menos 2 variáveis numéricas válidas.")
                st.stop()
                
            valid_targets = [c for c in all_numeric_cols if df[c].nunique() > 1]
            target_col = st.selectbox("2. Selecione a Variável Y (Dependente)", valid_targets)
            
            independent_cols = [c for c in all_numeric_cols if c != target_col]
            run_btn = st.button("🚀 Rodar Análise", use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

# Execução do Pipeline Analítico
if uploaded_file and 'run_btn' in locals() and run_btn:
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 1. Estatística Descritiva", 
        "📊 2. Distribuições", 
        "🔗 3. Correlação & Dispersão", 
        "🧮 4. Equação de Regressão Múltipla", 
        "📋 5. Diagnóstico Completo do Modelo"
    ])

    df_num = df[all_numeric_cols].dropna()
    
    reg_independent_cols = [c for c in independent_cols if df_num[c].nunique() > 1]
    const_independent_cols = [c for c in independent_cols if df_num[c].nunique() <= 1]
    
    X_multi = sm.add_constant(df_num[reg_independent_cols])
    Y = df_num[target_col]
    modelo_multi = sm.OLS(Y, X_multi).fit()
    
    corr_matrix = df_num.corr()
    
    # MÓDULO 1: Estatística Descritiva
    with tab1:
        st.header("Módulo 1: Estatística Descritiva Completa")
        desc_df = calcular_descritiva(df_num, all_numeric_cols)
        st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True)
        
        st.subheader("🔎 Interpretação Automática")
        valid_cv = desc_df.dropna(subset=['CV (%)'])
        if not valid_cv.empty:
            max_cv_var = valid_cv['CV (%)'].idxmax()
            max_cv_val = valid_cv['CV (%)'].max()
            st.markdown(f"**Heterogeneidade:** A variável com maior dispersão relativa é a **{max_cv_var}**, apresentando um Coeficiente de Variação de **{max_cv_val:.2f}%**.")
        
        st.markdown("**Análise de Assimetria:**")
        assimetrias = {c: analisar_assimetria(df_num[c]) for c in all_numeric_cols}
        for var, estado in assimetrias.items():
            st.markdown(f"- **{var}:** {estado}.")

    # MÓDULO 2: Frequência e Distribuição
    with tab2:
        st.header("Módulo 2: Distribuição de Frequências e Outliers")
        cols_ui = st.columns(2)
        for i, col in enumerate(all_numeric_cols):
            with cols_ui[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df_num[col], kde=True if df_num[col].nunique() > 1 else False, ax=ax, color='#3498DB')
                ax.set_title(f"Histograma: {col}")
                st.pyplot(fig)
                plt.close()
                
                shape = analisar_assimetria(df_num[col])
                Q1, Q3 = desc_df.loc[col, 'Q1'], desc_df.loc[col, 'Q3']
                IQR = Q3 - Q1
                outliers_abaixo = len(df_num[df_num[col] < (Q1 - 1.5 * IQR)]) if df_num[col].nunique() > 1 else 0
                outliers_acima = len(df_num[df_num[col] > (Q3 + 1.5 * IQR)]) if df_num[col].nunique() > 1 else 0
                
                with st.expander(f"Interpretação: {col}", expanded=True):
                    st.write(f"- **Forma:** {shape.lower()}.")
                    st.write(f"- **Outliers:** {outliers_abaixo} abaixo, {outliers_acima} acima.")

    # MÓDULO 3: Correlação e Dispersão
    with tab3:
        st.header("Módulo 3: Correlação de Pearson e Dispersão")
        
        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            targets_calculaveis = [c for c in all_numeric_cols if df_num[c].nunique() > 1]
            sns.heatmap(df_num[targets_calculaveis].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
            st.pyplot(fig)
            plt.close()
            
        with col2:
            corr_pairs = df_num[targets_calculaveis].corr().unstack().reset_index()
            corr_pairs.columns = ['Var1', 'Var2', 'Corr']
            corr_pairs = corr_pairs[corr_pairs['Var1'] != corr_pairs['Var2']]
            corr_pairs['Abs_Corr'] = corr_pairs['Corr'].abs()
            corr_pairs = corr_pairs.sort_values(by='Abs_Corr', ascending=False).drop_duplicates(subset=['Abs_Corr'])
            
            st.markdown("#### 🏆 Principais Correlações")
            top_pos = corr_pairs[corr_pairs['Corr'] > 0].head(3)
            top_neg = corr_pairs[corr_pairs['Corr'] < 0].head(3)
            
            st.write("**Top 3 Positivas:**")
            for _, r in top_pos.iterrows(): st.write(f"- {r['Var1']} & {r['Var2']}: **{r['Corr']:.3f}**")
            st.write("**Top 3 Negativas:**")
            for _, r in top_neg.iterrows(): st.write(f"- {r['Var1']} & {r['Var2']}: **{r['Corr']:.3f}**")
        
        st.markdown("---")
        st.subheader(f"Scatterplots (Impacto sobre Y: {target_col})")
        scatter_cols = st.columns(3)
        for i, indep in enumerate(independent_cols):
            with scatter_cols[i % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.regplot(data=df_num, x=indep, y=target_col, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'} if df_num[indep].nunique() > 1 else {'color':'none'})
                ax.set_title(f"Y vs {indep}")
                st.pyplot(fig)
                plt.close()

    # MÓDULO 4: EQUAÇÃO DE REGRESSÃO MÚLTIPLA
    with tab4:
        st.header("Módulo 4: Modelo Estimado Completo (Regressão Múltipla)")
        
        if const_independent_cols:
            st.warning(f"💡 **Nota sobre as Variáveis:** A variável **{', '.join(const_independent_cols)}** possui valor constante e foi absorvida.")

        intercepto = modelo_multi.params['const']
        partes_equacao = [f"{intercepto:.4f}"]
        
        for col in reg_independent_cols:
            coef = modelo_multi.params[col]
            sinal = "+" if coef >= 0 else "-"
            nome_formatado = formatar_texto_latex(col)
            partes_equacao.append(f"{sinal} ({abs(coef):.4f} \\cdot {nome_formatado})")
            
        equacao_completa_texto = " ".join(partes_equacao)
        target_formatado = formatar_texto_latex(target_col)
        
        st.info("### 🧮 Equação Geral Estimada:")
        st.write(f"$$ \\widehat{{{target_formatado}}} = {equacao_completa_texto} $$")
        
        st.markdown("---")
        st.subheader("📋 Significado Prático e Gerencial de Cada Coeficiente")
        dados_interpretacao = []
        for col in reg_independent_cols:
            coef = modelo_multi.params[col]
            p_val = modelo_multi.pvalues[col]
            sig = "Sim (Efeito Real)" if p_val < 0.05 else "Não (Ruído Estatístico)"
            sentido = "aumenta" if coef > 0 else "diminui"
            
            dados_interpretacao.append({
                "Variável Independente (X)": col,
                "Coeficiente (Impacto)": f"{coef:.4f}",
                "Significativo (Alfa 5%)": sig,
                "Interpretação Econômica/Operacional": f"Mantendo os demais fatores constantes, o incremento de 1 unidade em '{col}' causa uma variação média que {sentido} '{target_col}' em {abs(coef):.4f} unidades."
            })
            
        st.dataframe(pd.DataFrame(dados_interpretacao).set_index("Variável Independente (X)"), use_container_width=True)

    # MÓDULO 5: Diagnóstico e Validação Completa
    with tab5:
        st.header("Módulo 5: Diagnóstico, ANOVA e Resumos de Validação")
        st.text(modelo_multi.summary().tables[0].as_text())
        st.text(modelo_multi.summary().tables[1].as_text())
        
        st.markdown("---")
        st.subheader("🔎 Métricas de Ajuste Global")
        st.markdown(f"**Poder de Explicação Combinado (R² Global):** {modelo_multi.rsquared:.4f} *(As variáveis explicam {modelo_multi.rsquared*100:.1f}% de '{target_col}')*")
        st.markdown(f"**R² Ajustado:** {modelo_multi.rsquared_adj:.4f}")
        
        pvalues = modelo_multi.pvalues.drop('const')
        significativas = pvalues[pvalues < 0.05].index.tolist()
        
        st.write(f"- 🟢 **Variáveis com impacto real comprovado:** {', '.join(significativas) if significativas else 'Nenhuma'}")
        
        corr_with_y = corr_matrix[target_col].drop(target_col)
        sinais_alterados = []
        for col in reg_independent_cols:
            if (corr_with_y[col] > 0 and modelo_multi.params[col] < 0) or (corr_with_y[col] < 0 and modelo_multi.params[col] > 0):
                sinais_alterados.append(str(col))
                
        if sinais_alterados:
            st.warning(f"⚠️ As variáveis **{', '.join(sinais_alterados)}** inverteram de sinal no modelo. Indício de multicolinearidade.")
