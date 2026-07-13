import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, gaussian_kde
from scipy.linalg import eigh
import re

# Importação segura apenas para os testes de adequação
try:
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

# Configuração da Página
st.set_page_config(page_title="Análise Estatística & Regressão", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2C3E50; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Plataforma de Análise Estatística Avançada")
st.markdown("Faça o upload da sua base de dados, selecione o tipo de análise desejada e obtenha um relatório acadêmico completo.")

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
    if sk > 0.5: return "Assimétrica Positiva (cauda à direita)"
    elif sk < -0.5: return "Assimétrica Negativa (cauda à esquerda)"
    return "Relativamente Simétrica"

def format_p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

def formatar_texto_latex(texto):
    txt = str(texto).replace('%', '\\%').replace('$', '\\$').replace('_', '\\_')
    return f"\\text{{{txt}}}"

def recuperar_nota_corrompida(val):
    val_str = str(val).strip()
    match = re.match(r'^2026[-/](\d{2})[-/](\d{2})', val_str)
    if match:
        mes = int(match.group(1))
        dia = int(match.group(2))
        return float(f"{dia}.{mes}") if dia <= 5 else float(f"{mes}.{dia}")
    
    match_br = re.match(r'^(\d{2})[-/](\d{2})[-/](2026|\d{2})', val_str)
    if match_br:
        d = int(match_br.group(1))
        m = int(match_br.group(2))
        if d <= 5: return float(f"{d}.{m}")
        if m <= 5: return float(f"{m}.{d}")
        
    limpo = val_str.replace(',', '.')
    limpo = re.sub(r'[^\d\.\-]+', '', limpo)
    try: return float(limpo) if limpo else np.nan
    except: return np.nan

# Rotação Varimax nativa em NumPy (Evita o bug da biblioteca externa)
def varimax_rotation(Phi, gamma=1.0, max_iter=500, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(max_iter):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and (d - d_old) / d < tol: break
    return np.dot(Phi, R)

# Interface Lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_file = st.file_uploader("1. Upload da Base de Dados", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            for col in df.columns:
                if str(col).lower() not in ['obs', 'obs.', 'id', 'identificação', 'unidade', 'região']:
                    df[col] = df[col].apply(recuperar_nota_corrompida)

            all_numeric_cols = [c for c in df.select_dtypes(include=np.number).columns.tolist() if str(c).lower() not in ['obs', 'obs.', 'id']]
            all_numeric_cols = [c for c in all_numeric_cols if df[c].notna().sum() > 0]

            if len(all_numeric_cols) < 2:
                st.error("A base precisa ter pelo menos 2 variáveis numéricas válidas.")
                st.stop()
            
            df_num = df[all_numeric_cols].dropna()
            
            st.markdown("---")
            tipo_analise = st.radio("2. Selecione a Análise Desejada", ["📈 Estatística Descritiva & Regressão Múltipla", "🧬 Análise Fatorial Exploratória (AFE)"])
            
            st.markdown("---")
            if "Regressão" in tipo_analise:
                valid_targets = [c for c in all_numeric_cols if df[c].nunique() > 1]
                target_col = st.selectbox("3. Selecione a Variável Y (Dependente)", valid_targets)
                independent_cols = [c for c in all_numeric_cols if c != target_col]
            else:
                opcoes_fa = [c for c in all_numeric_cols if df_num[c].nunique() > 1]
                independent_cols = st.multiselect("3. Selecione as Variáveis para a Fatoração", opcoes_fa, default=opcoes_fa)
                
            run_btn = st.button("🚀 Rodar Análise", use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

# Execução do Pipeline Analítico
if uploaded_file and 'run_btn' in locals() and run_btn:
    reg_independent_cols = [c for c in independent_cols if df_num[c].nunique() > 1]
    
    # ------------------ OPÇÃO 1: REGRESSÃO LINEAR MULTIPLA ------------------
    if "Regressão" in tipo_analise:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 1. Estatística Descritiva", "📊 2. Distribuições", "🔗 3. Correlação & Dispersão", "🧮 4. Equação de Regressão Múltipla", "📋 5. Diagnóstico do Modelo"])
        X_multi = sm.add_constant(df_num[reg_independent_cols])
        Y = df_num[target_col]
        modelo_multi = sm.OLS(Y, X_multi).fit()

        with tab1:
            st.header("Módulo 1: Estatística Descritiva Completa")
            desc_df = calcular_descritiva(df_num, all_numeric_cols)
            st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True)

        with tab2:
            st.header("Módulo 2: Distribuição de Frequências")
            cols_ui = st.columns(2)
            for i, col in enumerate(all_numeric_cols):
                with cols_ui[i % 2]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df_num[col], kde=True if df_num[col].nunique() > 1 else False, ax=ax, color='#3498DB')
                    st.pyplot(fig)
                    plt.close()

        with tab3:
            st.header("Módulo 3: Correlação de Pearson e Dispersão")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(df_num[[c for c in all_numeric_cols if df_num[c].nunique() > 1]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig)
            plt.close()

        with tab4:
            st.header("Módulo 4: Modelo Estimado Completo")
            intercepto = modelo_multi.params['const']
            partes_equacao = [f"{intercepto:.4f}"]
            for col in reg_independent_cols:
                coef = modelo_multi.params[col]
                partes_equacao.append(f"{'+' if coef >= 0 else '-'} ({abs(coef):.4f} \\cdot {formatar_texto_latex(col)})")
            st.info("### 🧮 Equação Geral Estimada:")
            st.write(f"$$ \\widehat{{{formatar_texto_latex(target_col)}}} = {' '.join(partes_equacao)} $$")

        with tab5:
            st.header("Módulo 5: Diagnóstico, ANOVA e Resumos de Validação")
            st.text(modelo_multi.summary().tables[0].as_text())
            st.text(modelo_multi.summary().tables[1].as_text())

    # ------------------ OPÇÃO 2: ANÁLISE FATORIAL EXPLORATÓRIA ------------------
    else:
        st.header("🧬 Análise Fatorial Exploratória (AFE)")
        
        if len(reg_independent_cols) < 3:
            st.warning("É preciso selecionar pelo menos 3 colunas válidas para realizar uma fatoração consistente.")
        else:
            df_fa = df_num[reg_independent_cols].dropna().astype(float)
            
            tab_fa1, tab_fa2, tab_fa3 = st.tabs(["📋 1. Adequabilidade da Amostra", "📐 2. Autovalores (Scree Plot)", "📊 3. Cargas Fatoriais Rotacionadas"])
            
            with tab_fa1:
                st.subheader("Validação Estatística da Matriz de Correlações")
                if not FA_AVAILABLE:
                    st.warning("Testes KMO/Bartlett indisponíveis. Usando validação via Matriz de Correlação.")
                else:
                    try:
                        chi_square, p_value_bartlett = calculate_bartlett_sphericity(df_fa)
                        kmo_all, kmo_model = calculate_kmo(df_fa)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric(label="Coeficiente KMO Geral", value=f"{kmo_model:.3f}")
                            if kmo_model >= 0.6: st.success("✓ KMO adequado para fatoração.")
                            else: st.warning("⚠️ KMO baixo. Relações fracas entre variáveis.")
                        with c2:
                            st.metric(label="Teste de Bartlett (p-valor)", value=format_p_value(p_value_bartlett))
                            if p_value_bartlett < 0.05: st.success("✓ Dados significativamente correlacionados.")
                    except Exception as e:
                        st.error(f"Não foi possível processar KMO/Bartlett devido aos dados: {e}")
                    
            with tab_fa2:
                st.subheader("Critério de Kaiser (Eigenvalues > 1.0)")
                try:
                    # Cálculo matemático robusto e puro via Matriz de Correlação do Pandas e Álgebra Linear do SciPy
                    corr_matrix = df_fa.corr().values
                    ev, _ = eigh(corr_matrix)
                    ev = sorted(ev, reverse=True) # Ordena do maior autovalor para o menor
                    
                    n_fatores_sugeridos = max(1, sum(1 for x in ev if x >= 1.0))
                    
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.scatter(range(1, len(ev) + 1), ev, color='#E74C3C', zorder=3)
                    ax.plot(range(1, len(ev) + 1), ev, color='#34495E', linestyle='--')
                    ax.axhline(y=1, color='gray', linestyle=':')
                    ax.set_title("Gráfico de Sedimentação (Scree Plot)")
                    ax.set_xlabel("Fatores")
                    ax.set_ylabel("Autovalores")
                    st.pyplot(fig)
                    plt.close()
                    
                    st.info(f"💡 O algoritmo identificou **{n_fatores_sugeridos} fator(es)** com Autovalores $\\ge 1.0$.")
                except Exception as e:
                    st.error(f"Erro no cálculo dos Autovalores: {e}")
                    n_fatores_sugeridos = 1
                    
            with tab_fa3:
                st.subheader("Matriz de Cargas Fatoriais Rotacionada (Varimax)")
                try:
                    # Extração de componentes principais nativos via Decomposição em Valores Singulares (SVD)
                    A = df_fa.values - np.mean(df_fa.values, axis=0)
                    _, _, Vht = np.linalg.svd(A, full_matrices=False)
                    cargas_iniciais = Vht[:n_fatores_sugeridos].T * np.sqrt(ev[:n_fatores_sugeridos])
                    
                    # Aplica a rotação Varimax pura desenvolvida em NumPy
                    if n_fatores_sugeridos > 1:
                        cargas_rotacionadas = varimax_rotation(cargas_iniciais)
                    else:
                        cargas_rotacionadas = cargas_iniciais
                        
                    colunas_fatores = [f"Fator {i+1}" for i in range(n_fatores_sugeridos)]
                    df_cargas = pd.DataFrame(cargas_rotacionadas, columns=colunas_fatores, index=reg_independent_cols)
                    
                    st.dataframe(df_cargas.style.format("{:.3f}").background_gradient(cmap="bwr", vmin=-1, vmax=1), use_container_width=True)
                    st.markdown("**Interpretação acadêmica:** Cargas próximas a +1 ou -1 em um mesmo fator indicam que as variáveis compartilham a mesma dimensão conceitual oculta.")
                except Exception as e:
                    st.error(f"Erro ao extrair cargas fatoriais: {e}")
