import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, gaussian_kde
from scipy.signal import find_peaks

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(page_title="Análise Estatística & Regressão", layout="wide", page_icon="📊")

# Customização do estilo
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #2C3E50; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Plataforma de Análise Estatística e Regressão Linear")
st.markdown("Faça o upload da sua base de dados, defina as variáveis e obtenha uma análise acadêmica completa automatizada.")

# ==========================================
# Funções Auxiliares de Análise
# ==========================================
def calcular_descritiva(df, cols):
    stats = []
    for c in cols:
        s = df[c].dropna()
        mean = s.mean()
        med = s.median()
        mode_val = s.mode()
        std = s.std()
        stats.append({
            'Variável': c,
            'Média': mean,
            'Mediana': med,
            'Moda': mode_val.iloc[0] if not mode_val.empty else np.nan,
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
    sk = skew(s.dropna())
    if sk > 0.5:
        return "Assimétrica Positiva (cauda à direita)"
    elif sk < -0.5:
        return "Assimétrica Negativa (cauda à esquerda)"
    return "Relativamente Simétrica"

def detectar_bimodalidade(s):
    s_clean = s.dropna()
    if len(s_clean) < 10: return False
    kde = gaussian_kde(s_clean)
    x_range = np.linspace(s_clean.min(), s_clean.max(), 100)
    peaks, _ = find_peaks(kde(x_range))
    return len(peaks) > 1

def format_p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"

# ==========================================
# Interface Lateral (Sidebar)
# ==========================================
with st.sidebar:
    st.header("⚙️ Configurações do Modelo")
    uploaded_file = st.file_uploader("1. Upload da Base de Dados", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("A base precisa ter pelo menos 2 variáveis numéricas.")
                st.stop()

            target_col = st.selectbox("2. Selecione a Variável Y (Dependente)", numeric_cols)
            independent_cols = [c for c in numeric_cols if c != target_col]
            run_btn = st.button("🚀 Rodar Análise", use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

# ==========================================
# Execução do Pipeline Analítico
# ==========================================
if uploaded_file and 'run_btn' in locals() and run_btn:

    # Abas do aplicativo
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 1. Estatística Descritiva",
        "📊 2. Distribuições",
        "🔗 3. Correlação & Dispersão",
        "📉 4. Regressão Simples",
        "📉 5. Regressão Múltipla"
    ])

    df_num = df[numeric_cols].dropna() # Trabalhando com dados completos para regressão

    # ---------------------------------------------------------
    # MÓDULO 1: Estatística Descritiva
    # ---------------------------------------------------------
    with tab1:
        st.header("Módulo 1: Estatística Descritiva Completa")
        desc_df = calcular_descritiva(df_num, numeric_cols)
        st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True)

        st.subheader("🔎 Interpretação Automática")

        # Maior Heterogeneidade
        max_cv_var = desc_df['CV (%)'].idxmax()
        max_cv_val = desc_df['CV (%)'].max()

        st.markdown(f"**Heterogeneidade:** A variável com maior dispersão relativa é a **{max_cv_var}**, apresentando um Coeficiente de Variação de **{max_cv_val:.2f}%**.")
        st.info(f"**Significado Gerencial:** Uma alta variação em `{max_cv_var}` sugere baixa padronização operacional (ex: na gestão de uma rede como a *Atlas Hotéis*, isso pode indicar inconsistência entre unidades, dependência excessiva de sazonalidade ou flutuações de mercado não controladas).")

        # Assimetria
        st.markdown("**Análise de Assimetria:**")
        assimetrias = {c: analisar_assimetria(df_num[c]) for c in numeric_cols}
        for var, estado in assimetrias.items():
            if "Assimétrica" in estado:
                st.markdown(f"- **{var}:** {estado}.")

        st.info("**Significado Gerencial da Assimetria:** Assimetria positiva indica uma concentração de valores baixos com raros picos muito altos (ex: algumas poucas reservas de altíssimo valor puxando a média). Assimetria negativa aponta uma concentração em valores altos, com raras quedas bruscas (ex: taxa de ocupação quase sempre cheia, mas com quedas raras que formam a cauda).")

    # ---------------------------------------------------------
    # MÓDULO 2: Frequência e Distribuição
    # ---------------------------------------------------------
    with tab2:
        st.header("Módulo 2: Distribuição de Frequências e Outliers")

        # Variáveis solicitadas + Fallback se não existirem com esses nomes exatos
        esperadas = ["Taxa de Ocupação", "Investimento em Marketing Digital", "Avaliação Online", "Diária Média"]
        plot_cols = [c for c in esperadas if c in df_num.columns]
        if not plot_cols:
            plot_cols = numeric_cols[:4] # Plota as 4 primeiras se os nomes não baterem

        cols_ui = st.columns(2)
        for i, col in enumerate(plot_cols):
            with cols_ui[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df_num[col], kde=True, ax=ax, color='#3498DB')
                ax.set_title(f"Histograma: {col}")
                st.pyplot(fig)
                plt.close()

                # Análise da forma e outliers
                shape = analisar_assimetria(df_num[col])
                bimodal = detectar_bimodalidade(df_num[col])
                if bimodal: shape = "Bimodal (possível existência de dois grupos distintos)"

                Q1, Q3 = desc_df.loc[col, 'Q1'], desc_df.loc[col, 'Q3']
                IQR = Q3 - Q1
                outliers_abaixo = len(df_num[df_num[col] < (Q1 - 1.5 * IQR)])
                outliers_acima = len(df_num[df_num[col] > (Q3 + 1.5 * IQR)])

                with st.expander(f"Interpretação: {col}", expanded=True):
                    st.write(f"- **Forma:** {shape.lower()}.")
                    st.write(f"- **Outliers:** {outliers_abaixo} abaixo do limite inferior, {outliers_acima} acima do limite superior.")

                    if outliers_acima > 0 or outliers_abaixo > 0:
                        st.write("💡 *Visão Gerencial:* Outliers nem sempre são erros. Valores discrepantes acima da média em 'Avaliações' ou 'Ocupação' representam casos de extremo sucesso (oportunidades para clonar boas práticas). Já outliers abaixo apontam gargalos críticos urgentes.")

    # ---------------------------------------------------------
    # MÓDULO 3: Correlação e Dispersão
    # ---------------------------------------------------------
    with tab3:
        st.header("Módulo 3: Correlação de Pearson e Dispersão")

        corr_matrix = df_num.corr()

        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
            st.pyplot(fig)
            plt.close()

        with col2:
            # Organizando pares de correlação
            corr_pairs = corr_matrix.unstack().reset_index()
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

            multicolinearidade = corr_pairs[corr_pairs['Abs_Corr'] > 0.85]
            if not multicolinearidade.empty:
                st.warning(f"⚠️ **Multicolinearidade detectada (|r| > 0.85):** Pode causar redundância no modelo múltiplo.")
                for _, r in multicolinearidade.iterrows(): st.write(f"- {r['Var1']} e {r['Var2']} ({r['Corr']:.2f})")

        # Dispersão
        st.markdown("---")
        st.subheader(f"Scatterplots (Impacto sobre Y: {target_col})")

        corr_with_y = corr_matrix[target_col].drop(target_col)
        best_x = corr_with_y.abs().idxmax()

        st.info(f"**Atenção:** A variável independente com maior correlação com '{target_col}' é **'{best_x}'** (r = {corr_with_y[best_x]:.3f}). Confirme se isso se alinha com a literatura e as expectativas de negócio ou se é uma correlação espúria.")

        scatter_cols = st.columns(3)
        for i, indep in enumerate(independent_cols):
            with scatter_cols[i % 3]:
                fig, ax = plt.subplots(figsize=(5, 4))
                # Modified line: Pass .values to ensure array-like input
                sns.regplot(x=df_num[indep].values, y=df_num[target_col].values, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
                ax.set_title(f"Y vs {indep}")
                st.pyplot(fig)
                plt.close()

                r_val = corr_with_y[indep]
                relacao = "Forte" if abs(r_val) >= 0.7 else "Moderada" if abs(r_val) >= 0.4 else "Fraca/Inexistente"
                st.caption(f"**Relação Linear:** {relacao} (r={r_val:.2f})")

    # ---------------------------------------------------------
    # MÓDULO 4: Regressão Simples
    # ---------------------------------------------------------
    with tab4:
        st.header("PARTE IV: 4.1 Regressão Linear Simples")
        st.markdown(f"Modelo selecionado automaticamente usando o melhor preditor linear: **{best_x}**")

        X_simples = sm.add_constant(df_num[best_x])
        Y = df_num[target_col]
        modelo_simples = sm.OLS(Y, X_simples).fit()

        b0 = modelo_simples.params['const']
        b1 = modelo_simples.params[best_x]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Equação Estimada\n$$ Y = {b0:.4f} + ({b1:.4f} \cdot X) $$")
            st.markdown(f"**Onde:** Y = {target_col}, X = {best_x}")

            st.markdown(f"""
            **Interpretação Gerencial:**
            * **Intercepto ($b_0 = {b0:.2f}$):** Valor teórico esperado de {target_col} quando {best_x} for estritamente zero.
            * **Coeficiente ($b_1 = {b1:.2f}$):** Para cada unidade adicional de {best_x}, espera-se uma variação média de {b1:.2f} em {target_col}.
            """)

        with col2:
            st.markdown("### Qualidade e Testes de Hipótese")
            r2_simples = modelo_simples.rsquared
            st.write(f"- **R²:** {r2_simples:.4f} *(O modelo explica {r2_simples*100:.1f}% da variação de {target_col})*")

            f_stat = modelo_simples.fvalue
            f_pval = modelo_simples.f_pvalue
            st.write(f"- **Teste F (Global):** F = {f_stat:.2f} | p-valor = {format_p_value(f_pval)}")

            t_stat = modelo_simples.tvalues[best_x]
            t_pval = modelo_simples.pvalues[best_x]
            st.write(f"- **Teste t (Variável {best_x}):** t = {t_stat:.2f} | p-valor = {format_p_value(t_pval)}")

            if t_pval < 0.05:
                st.success("✅ O coeficiente angular é estatisticamente significativo a 5%. A variável tem impacto real.")
            else:
                st.error("❌ O coeficiente angular não é significativo a 5%. A relação pode ser fruto do acaso.")

    # ---------------------------------------------------------
    # MÓDULO 5: Regressão Múltipla
    # ---------------------------------------------------------
    with tab5:
        st.header("PARTE IV: 4.2 Regressão Linear Múltipla")

        X_multi = sm.add_constant(df_num[independent_cols])
        modelo_multi = sm.OLS(Y, X_multi).fit()

        st.markdown("### Resumo Estatístico do Modelo (Statsmodels Style)")
        st.text(modelo_multi.summary().tables[0].as_text())
        st.text(modelo_multi.summary().tables[1].as_text())

        r2_m = modelo_multi.rsquared
        r2_adj = modelo_multi.rsquared_adj

        st.markdown("---")
        st.subheader("Análise Avançada e Interpretação")

        st.markdown(f"**Comparação R²:** O R² Simples era de **{r2_simples:.4f}**. No modelo Múltiplo, o R² foi para **{r2_m:.4f}** e o Ajustado para **{r2_adj:.4f}**. O R² Ajustado penaliza a adição de variáveis que não melhoram o poder preditivo, sendo o indicador mais confiável neste cenário.")

        # Teste F Global
        st.markdown(f"**Teste F (ANOVA):** F = {modelo_multi.fvalue:.2f}, p-valor: {format_p_value(modelo_multi.f_pvalue)}.")
        if modelo_multi.f_pvalue < 0.05:
            st.success("✅ Teste F Global: Ao menos uma das variáveis independentes possui efeito significativo sobre Y (Modelo globalmente válido a 5%).")
        else:
            st.error("❌ Teste F Global: O modelo inteiro falha em prever a variável dependente significativamente melhor que a média simples.")

        st.markdown("**Significância Individual (Alfa 5%):**")
        pvalues = modelo_multi.pvalues.drop('const')
        significativas = pvalues[pvalues < 0.05].index.tolist()
        nao_significativas = pvalues[pvalues >= 0.05].index.tolist()

        st.write(f"- 🟢 **Variáveis Significativas:** {', '.join(significativas) if significativas else 'Nenhuma'}")
        st.write(f"- 🔴 **Variáveis Não-Significativas:** {', '.join(nao_significativas) if nao_significativas else 'Nenhuma'}")

        # Sinais inesperados vs correlação original
        st.markdown("**Detecção de Sinais Inesperados (Supressão/Multicolinearidade):**")
        sinais_alterados = []
        for col in independent_cols:
            corr_orig = corr_with_y[col]
            coef_multi = modelo_multi.params[col]
            if (corr_orig > 0 and coef_multi < 0) or (corr_orig < 0 and coef_multi > 0):
                sinais_alterados.append(col)

        if sinais_alterados:
            st.warning(f"⚠️ Atenção! As variáveis **{', '.join(sinais_alterados)}** trocaram de sinal (positivo vs negativo) da correlação simples para o coeficiente na regressão múltipla. Isso é um forte indício de multicolinearidade ou efeito de variáveis de confusão (Simpson's Paradox).")
        else:
            st.info("✓ Todos os sinais dos coeficientes no modelo múltiplo acompanham o sentido das suas correlações individuais.")
