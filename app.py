"""Previsão de término de obras (AFT Log-Normal) em Streamlit."""
from __future__ import annotations

import io
import os
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import aft_core as core

MARCOS = core.MARCOS_DIAS  # (0, 30, 60): dias depois do prazo original

# Para pré-carregar um CSV sem usar o upload (útil rodando localmente):
#   PREVISAO_CSV=caminho/contratos.csv streamlit run app.py
CSV_PRE_CARREGADO = os.environ.get("PREVISAO_CSV", "").strip()

# Cores (checadas com o validador da skill dataviz; funcionam em tema claro e escuro).
COR_CURVA = "#3987e5"                                         # série única: azul
COR_REFERENCIA = "#898781"                                    # linhas de referência e rótulos (neutro)
COR_DESTAQUE = "#eb6834"                                      # o resultado real
COR_LINHAS = {                                                # linhas verticais do gráfico: uma cor por limite
    "Prazo original": "#8e44ad",
    "Prazo aditivado": "#c79a00",
    "Hoje": "#6b7280",
    "Mediana (50%)": "#2a9d55",
    "Término real": COR_DESTAQUE,
}

# Explicações dos conceitos: aparecem no ícone (?) de cada título, para a tela ficar limpa.
AJUDA = {
    "probabilidade": (
        "Chance de a obra terminar **até** a data indicada, no dia ou antes. Cada cartão inclui os anteriores, "
        "por isso a porcentagem só cresce. *Exemplo:* 70% quer dizer que, em 10 obras parecidas com esta, "
        "cerca de 7 terminariam até essa data."
    ),
    "condicional": (
        "Todas as porcentagens partem do fato de que a obra **ainda não terminou hoje**. "
        "Datas que já passaram sem a obra terminar valem 0%."
    ),
    "mediana": (
        "Metade das chances: em 50% dos casos a obra termina até esta data e, nos outros 50%, depois. "
        "É o valor “típico” (a mediana)."
    ),
    "pessimista": (
        "Em 80% dos casos a obra termina até esta data; só 1 obra em 5 passaria dela. "
        "Serve como cenário com folga, para planejar com segurança."
    ),
    "faixa80": "Intervalo em que o modelo espera o término em 80% dos casos: 10% terminam antes e 10% depois.",
    "curva": "Para cada data, a chance de a obra já ter terminado. Sobe até chegar perto de 100%.",
    "faixa": (
        "As faixas são fatias de tempo depois do prazo original, que não se sobrepõem: "
        "*até o prazo*, *1 a 30 dias depois*, *31 a 60 dias depois* e assim por diante."
    ),
    "epoca": (
        "A previsão usa só o que se sabia naquele dia: obras concluídas depois dele não entram no modelo. "
        "Assim a comparação com o que de fato aconteceu é honesta."
    ),
    "limites": (
        "O modelo aprende com algumas dezenas de obras e usa só o prazo original. "
        "Não conhece chuvas, fornecedores nem aditivos em negociação. Obras em andamento entram como "
        "“ainda não terminou” e as suspensas ficam de fora. Use como apoio para priorizar, não como compromisso de data."
    ),
}

st.set_page_config(page_title="Previsão de término de obras", layout="wide")

# Garante que o app e o núcleo são da mesma versão (evita um KeyError difícil de entender após um deploy parcial).
VERSAO_ESPERADA = "2026-11-13"
if getattr(core, "VERSAO_NUCLEO", None) != VERSAO_ESPERADA:
    st.error(
        "O arquivo **aft_core.py** carregado está desatualizado em relação ao app.py. "
        f"Arquivo em uso: `{core.__file__}`. Envie o aft_core.py novo ao repositório (junto com o app.py) "
        "e depois use *Manage app → Reboot app*."
    )
    st.stop()

# Datas e rótulos longos cortavam nos cartões (st.metric); reduz a fonte do valor e deixa o rótulo quebrar linha.
st.markdown(
    """
<style>
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Dados e modelo (em cache: o ajuste leva milissegundos, mas evita refazer a cada clique)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def ler_csv(conteudo: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(conteudo), encoding="utf-8-sig", sep=None, engine="python")


@st.cache_data(show_spinner=False)
def ajustar(contratos: pd.DataFrame, data_corte: date, alpha: float, versao_nucleo: str):
    df_prep = core.preparar_contratos(contratos, data_corte)
    modelo = core.treinar_modelo(df_prep, alpha)
    return df_prep, modelo


@st.cache_data(show_spinner=False)
def modelo_da_epoca(df_prep: pd.DataFrame, cutoff: date, alpha: float, versao_nucleo: str):
    return core.treinar_no_passado(df_prep, cutoff, alpha)


# --------------------------------------------------------------------------- #
# Formatação
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    if x <= 0:
        return "0%"
    if x >= 1:
        return "100%"
    if x < 0.005:
        return "<1%"
    if x > 0.995:
        return ">99%"
    return f"{100 * x:.0f}%"


def fmt_data(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def fmt_dias(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".") + " dias"


def soma_dias(data_os, dias: float) -> pd.Timestamp:
    return pd.Timestamp(data_os) + pd.Timedelta(days=int(round(dias)))


def n_dias(n: float) -> str:
    n = int(round(abs(n)))
    return f"{n} dia" if n == 1 else f"{n} dias"


def titulo_marco(offset: int) -> str:
    return "Até o prazo original" if offset == 0 else f"Até {offset} dias depois"


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def grafico_curva(modelo, data_os, prazo, vigente, decorridos, p50_dias, p90_dias, real_dias=None):
    """Curva: probabilidade de a obra já ter terminado em cada data, com marcos de referência."""
    ja_rel = decorridos / prazo
    x_max = max(p90_dias * 1.1, vigente * 1.25, decorridos * 1.1, prazo * 1.5)
    if real_dias:
        x_max = max(x_max, real_dias * 1.1)
    dias = np.linspace(max(decorridos, 1), x_max, 160)
    cdf = modelo.cdf_condicional(prazo, ja_rel, dias / prazo)
    curva = pd.DataFrame({
        "data": pd.Timestamp(data_os) + pd.to_timedelta(dias, unit="D"),
        "prob": cdf,
    })

    marcas = [("Prazo original", prazo, [6, 4])]
    if vigente > prazo:
        marcas.append(("Prazo aditivado", vigente, [2, 3]))
    if decorridos > 0:
        marcas.append(("Hoje", decorridos, [1, 0]))
    marcas.append(("Mediana (50%)", p50_dias, [8, 2, 2, 2]))
    if real_dias:
        marcas.append(("Término real", real_dias, [1, 0]))

    rot = [m[0] for m in marcas]
    df_m = pd.DataFrame({
        "rotulo": rot,
        "data": [pd.Timestamp(data_os) + pd.Timedelta(days=float(m[1])) for m in marcas],
        "y": [max(0.96 - 0.085 * i, 0.05) for i in range(len(marcas))],
    })
    meio = curva["data"].iloc[len(curva) // 2]
    df_m["lado"] = np.where(df_m["data"] > meio, "dir", "esq")

    cores = [COR_LINHAS[r] for r in rot]
    tracos = [m[2] for m in marcas]

    x_enc = alt.X("data:T", title=None, axis=alt.Axis(format="%d/%m/%y", tickCount=6, labelOverlap=True))
    y_enc = alt.Y("prob:Q", title="Prob. de já ter terminado", axis=alt.Axis(format="%"),
                  scale=alt.Scale(domain=[0, 1]))

    linha = alt.Chart(curva).mark_line(strokeWidth=2, color=COR_CURVA).encode(x=x_enc, y=y_enc)
    pontos = alt.Chart(curva).mark_point(opacity=0, size=140).encode(
        x="data:T", y="prob:Q",
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                 alt.Tooltip("prob:Q", title="Prob. de já ter terminado", format=".0%")],
    )
    regras = alt.Chart(df_m).mark_rule(strokeWidth=1.5).encode(
        x="data:T",
        color=alt.Color("rotulo:N", scale=alt.Scale(domain=rot, range=cores), legend=None),
        strokeDash=alt.StrokeDash("rotulo:N", scale=alt.Scale(domain=rot, range=tracos), legend=None),
        tooltip=[alt.Tooltip("rotulo:N", title=""), alt.Tooltip("data:T", title="Data", format="%d/%m/%Y")],
    )
    camadas = [linha, pontos, regras]
    for lado, align, dx in (("esq", "left", 5), ("dir", "right", -5)):
        sub = df_m[df_m["lado"] == lado]
        if len(sub):
            camadas.append(
                alt.Chart(sub).mark_text(align=align, baseline="middle", dx=dx, fontSize=11, color=COR_REFERENCIA)
                .encode(x="data:T", y=alt.Y("y:Q", title="Prob. de já ter terminado", axis=alt.Axis(format="%"),
                                           scale=alt.Scale(domain=[0, 1])), text="rotulo:N")
            )
    return alt.layer(*camadas).properties(height=320)


# --------------------------------------------------------------------------- #
# Painel de previsão (usado no simulador e na aba de contratos reais)
# --------------------------------------------------------------------------- #
def titulo_marco_adit(offset: int) -> str:
    return "Até o prazo aditivado" if offset == 0 else f"Até {offset} dias depois do prazo aditivado"


def linha_marcos(itens, data_os, decorridos, titulo):
    """Uma linha de cartões com a probabilidade acumulada em cada marco."""
    for col, it in zip(st.columns(len(itens)), itens):
        col.metric(titulo(it["offset"]), fmt_pct(it["prob_ate"]))
        passou = decorridos >= it["limite_dias"]
        col.caption(fmt_data(soma_dias(data_os, it["limite_dias"])) + ("  ·  já passou" if passou else ""))


def painel_previsao(modelo, data_os, prazo, vigente, decorridos, real_dias=None):
    """Mostra marcos, datas-chave e curva. `vigente=None` = sem aditivo conhecido. Devolve o dict da previsão."""
    prazo = float(prazo)
    vigente_dias = float(vigente) if vigente else prazo
    p = core.previsao_contrato(modelo, prazo, decorridos, vigente_dias, MARCOS)
    adit = int(round(vigente_dias - prazo)) if p["vigente"] else 0

    st.subheader("Probabilidade de a obra terminar…", help=AJUDA["probabilidade"] + "\n\n" + AJUDA["condicional"])
    if p["vigente"]:
        st.caption("**No prazo original**")
    linha_marcos(p["original"], data_os, decorridos, titulo_marco)
    if p["vigente"]:
        st.markdown("---")
        st.caption(f"**No prazo aditivado** (prazo original + {adit} dias de aditivo)")
        linha_marcos(p["vigente"], data_os, decorridos, titulo_marco_adit)

    st.divider()
    q = p["quantis"]
    st.subheader("Quando deve terminar", help=AJUDA["condicional"] + "\n\n" + AJUDA["curva"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Duração (mediana)", fmt_dias(q["p50"]), help=AJUDA["mediana"])
    m1.caption(f"{q['p50'] / prazo:.2f}× o prazo original")
    m2.metric("Término (mediana)", fmt_data(soma_dias(data_os, q["p50"])), help=AJUDA["mediana"])
    m3.metric("Término (pessimista)", fmt_data(soma_dias(data_os, q["p80"])), help=AJUDA["pessimista"])
    st.caption(
        f"Em 80% dos casos, entre **{fmt_data(soma_dias(data_os, q['p10']))}** e "
        f"**{fmt_data(soma_dias(data_os, q['p90']))}**.",
        help=AJUDA["faixa80"],
    )

    st.divider()
    st.altair_chart(grafico_curva(modelo, data_os, prazo, vigente_dias, decorridos, q["p50"], q["p90"], real_dias))
    return p


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Dados")
    arquivo = st.file_uploader(
        "CSV de contratos", type="csv",
        help="Os dados enviados ficam só na memória da sessão; nada é gravado no repositório.",
    )
    if arquivo is not None:
        contratos = ler_csv(arquivo.getvalue())
    elif CSV_PRE_CARREGADO and Path(CSV_PRE_CARREGADO).is_file():
        contratos = ler_csv(Path(CSV_PRE_CARREGADO).read_bytes())
        st.caption(f"CSV pré-carregado: {Path(CSV_PRE_CARREGADO).name}")
    else:
        st.info("Envie o CSV de contratos para começar.")
        st.stop()

    data_corte = st.date_input(
        "Data de hoje (corte)",
        value=date.today(),
        format="DD/MM/YYYY",
        help="Obras sem conclusão até esta data são tratadas como “ainda em andamento”.",
    )

    if {"data_fim_observacao", "evento_conclusao"} <= set(contratos.columns):
        # a base já traz a data de corte com que foi gerada (a dos contratos em andamento)
        _cortes = pd.to_datetime(
            contratos.loc[contratos["evento_conclusao"].eq(0), "data_fim_observacao"], errors="coerce"
        ).dropna().unique()
        if len(_cortes) == 1 and pd.Timestamp(_cortes[0]).date() != data_corte:
            st.caption(f"Esta base foi gerada com corte em {fmt_data(_cortes[0])}.")

    with st.expander("Avançado"):
        alpha = st.select_slider(
            "Regularização (alpha)", options=core.ALPHAS, value=core.ALPHA_PADRAO,
            help="Quanto maior, mais o modelo evita se ajustar ao acaso dos poucos contratos. Na dúvida, não mexa.",
        )

faltantes = core.colunas_faltantes(contratos)
if faltantes:
    st.error("O CSV não tem as colunas obrigatórias: " + ", ".join(faltantes))
    st.stop()

df_prep, modelo = ajustar(contratos, data_corte, alpha, core.VERSAO_NUCLEO)
em_andamento = core.prever_em_andamento(df_prep, modelo)

n_invalidos = int((~df_prep["flag_apto_modelagem"].astype(bool)).sum()) - int(
    df_prep["grupo_modelagem"].eq("SUSPENSO_PARALISADO").sum()
)

# --------------------------------------------------------------------------- #
# Cabeçalho
# --------------------------------------------------------------------------- #
st.title("Previsão de término de obras", help=AJUDA["limites"])
if n_invalidos > 0:
    st.caption(f"{n_invalidos} linha(s) sem data de OS ou prazo válido foram ignoradas.")
if not modelo.convergiu:
    st.warning("O otimizador não convergiu. Revise os dados antes de usar os resultados.")

aba_sim, aba_real, aba_and = st.tabs(["Simulador", "Contratos reais", "Contratos em andamento"])

# --------------------------------------------------------------------------- #
# Aba 1 — Simulador
# --------------------------------------------------------------------------- #
with aba_sim:
    esq, dir_ = st.columns([1, 2], gap="large")

    with esq:
        st.subheader("Contrato")
        data_os_sim = st.date_input(
            "Data da OS (início da obra)", value=data_corte, key="sim_os", format="DD/MM/YYYY",
            help="Dia em que a ordem de serviço autorizou o início da obra.",
        )
        prazo = st.number_input(
            "Prazo original (dias)", 30, 3650, 360, step=30, key="sim_prazo",
            help="Prazo da ordem de serviço, antes de qualquer aditivo.",
        )
        total_adit = int(st.number_input(
            "Aditivos de prazo (dias)", 0, 3650, 0, step=30, key="sim_adit_dias",
            help="Total de dias acrescentados por aditivos. *Exemplo:* 60 + 30 dias de aditivo = 90. "
                 "Deixe 0 se não houve aditivo; com aditivo, o app mostra também as probabilidades contadas do prazo aditivado.",
        ))
        vigente = int(prazo) + total_adit

        st.divider()
        st.subheader("Situação hoje")
        hoje_sim = st.date_input(
            "Data de hoje", value=data_corte, key="sim_hoje", format="DD/MM/YYYY",
            help="Dia da previsão. O app conta quantos dias se passaram desde a OS e considera que a obra "
                 "**ainda não terminou** nesse dia. Se a obra não começou, use a mesma data da OS.",
        )
        decorridos = max((hoje_sim - data_os_sim).days, 0)
        if hoje_sim < data_os_sim:
            st.warning("A data de hoje é anterior à OS; considerando obra ainda não iniciada (0 dias).")

    with dir_:
        resumo = f"Término original **{fmt_data(soma_dias(data_os_sim, prazo))}**"
        if total_adit:
            resumo += f"  ·  com +{total_adit} dias de aditivo, **{fmt_data(soma_dias(data_os_sim, vigente))}**"
        st.caption(resumo + f"  ·  hoje é o **dia {decorridos}** ({100 * decorridos / prazo:.0f}% do prazo original)")

        if decorridos >= vigente:
            st.warning(
                "Hoje já passou do prazo aditivado e a obra não terminou: a probabilidade de estourar é de 100%."
                if total_adit else
                "Hoje já passou do prazo original e a obra não terminou: a probabilidade de estourar é de 100%."
            )
        elif decorridos >= prazo:
            st.info("Hoje já passou do prazo original, mas ainda está dentro do prazo aditivado.")

        st.divider()
        painel_previsao(modelo, data_os_sim, prazo, vigente if total_adit else None, decorridos)

# --------------------------------------------------------------------------- #
# Aba 2 — Contratos reais (seleção e verificação)
# --------------------------------------------------------------------------- #
with aba_real:
    reais = df_prep[df_prep["data_os"].notna() & (df_prep["prazo_original_dias"] > 0)].copy()
    filtro = st.radio("Mostrar", ["Todos", "Em andamento", "Concluídos"], horizontal=True, key="real_filtro")
    if filtro == "Em andamento":
        reais = reais[reais["grupo_modelagem"] != "CONCLUIDO_COM_TARGET"]
    elif filtro == "Concluídos":
        reais = reais[reais["grupo_modelagem"] == "CONCLUIDO_COM_TARGET"]
    reais = reais.sort_values("id_contrato")

    SITUACAO = {
        "CONCLUIDO_COM_TARGET": "concluído",
        "EM_ANDAMENTO_CENSURADO": "em andamento",
        "SUSPENSO_PARALISADO": "suspenso/paralisado",
    }

    def _txt(r, col):
        v = r.get(col)
        return "" if v is None or pd.isna(v) else str(v)

    def rotulo_contrato(r) -> str:
        partes = [str(r["id_contrato"])]
        for col in ("Numero_contrato", "Nome_predio", "classificacao_contrato"):
            if _txt(r, col):
                partes.append(_txt(r, col))
        partes += [f"OS {fmt_data(r['data_os'])}", SITUACAO[r["grupo_modelagem"]]]
        return " · ".join(partes)

    if reais.empty:
        st.info("Nenhum contrato neste filtro.")
    else:
        rotulos = {r["id_contrato"]: rotulo_contrato(r) for _, r in reais.iterrows()}
        ids = list(rotulos)
        concl_recentes = reais[reais["grupo_modelagem"] == "CONCLUIDO_COM_TARGET"].sort_values("data_os")
        idx_padrao = ids.index(concl_recentes.iloc[-1]["id_contrato"]) if len(concl_recentes) else 0
        escolhido = st.selectbox(
            "Contrato", ids, index=idx_padrao, format_func=lambda i: rotulos[i], key="real_contrato",
            help="Obra em andamento: mostra a previsão de hoje. Obra concluída: mostra a previsão feita no início "
                 "da obra ao lado do que de fato aconteceu.",
        )
        c = reais[reais["id_contrato"] == escolhido].iloc[0]

        data_os = c["data_os"]
        prazo_c = int(c["prazo_original_dias"])
        vigente_c = int(round(c["prazo_vigente_atual_dias"]))
        adit_c = int(round(c["dias_aditivados_app"]))
        da_tabela = bool(c["aditivos_da_tabela"])
        n_adit_c = _txt(c, "n_aditivos_prazo")
        grupo = c["grupo_modelagem"]
        concluido = grupo == "CONCLUIDO_COM_TARGET"
        status_txt = str(c.get("status_final_controle", "—"))

        if _txt(c, "Nome_predio") or _txt(c, "Numero_contrato"):
            st.caption(
                f"**{_txt(c, 'Nome_predio') or '—'}**  ·  contrato nº {_txt(c, 'Numero_contrato') or '—'}"
                + (f"  ·  {_txt(c, 'classificacao_contrato')}" if _txt(c, "classificacao_contrato") else "")
            )
        st.divider()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Ordem de serviço", fmt_data(data_os))
        a2.metric("Prazo original", fmt_dias(prazo_c))
        a2.caption(f"Término: {fmt_data(soma_dias(data_os, prazo_c))}")
        a3.metric("Aditivos de prazo", f"+{adit_c} dias" if adit_c else "Nenhum")
        if adit_c and da_tabela and n_adit_c:
            a3.caption(f"{int(float(n_adit_c))} aditivo(s)")
        a4.metric("Prazo vigente", fmt_dias(vigente_c))
        a4.caption(f"Término: {fmt_data(soma_dias(data_os, vigente_c))}")
        if da_tabela:
            if int(c.get("flag_dias_divergem_por_data", 0) or 0):
                por_data = c.get("dias_aditivados_por_data")
                st.warning(
                    f"A soma dos aditivos ({adit_c} dias) difere da diferença entre as datas"
                    + (f" ({int(por_data)} dias)" if por_data is not None and pd.notna(por_data) else "")
                    + ". O app usa a soma da tabela de aditivos."
                )
            if int(c.get("flag_aditivo_pos_observacao", 0) or 0):
                st.info("Parte do aditivo foi assinada depois da conclusão da obra, então não era conhecida durante a execução.")
        elif adit_c:
            st.caption("Este CSV traz só as datas: os dias aditivados são a diferença entre elas.")

        if concluido:
            conclusao = c["data_conclusao_definitiva"]
            dur = int(c["tempo_observado_dias"])
            if status_txt.upper().startswith("EM ANDAMENTO") or int(c.get("flag_status_desatualizado", 0) or 0):
                st.warning(
                    "No controle este contrato consta como EM ANDAMENTO, mas tem data de conclusão preenchida. "
                    "O modelo o trata como concluído."
                )
            if conclusao > pd.Timestamp(data_corte):
                st.warning("A data de conclusão registrada é posterior à data de corte.")

            # A previsão é feita no dia da OS, com o modelo "da época" (só o que se sabia naquele dia).
            dia = 0
            data_previsao = pd.Timestamp(data_os)
            m_ep, _info = modelo_da_epoca(df_prep, data_previsao.date(), alpha, core.VERSAO_NUCLEO)
            m_uso = m_ep if m_ep is not None else modelo

            p = core.previsao_contrato(m_uso, float(prazo_c), dia, None, MARCOS)
            q = p["quantis"]
            atraso = (pd.Timestamp(conclusao) - soma_dias(data_os, prazo_c)).days
            nome_faixa = p["faixas"][core.faixa_do_atraso(atraso, MARCOS)][0]

            st.divider()
            v1, v2, v3 = st.columns(3)
            v1.metric("Término real", fmt_data(conclusao))
            v1.caption(f"{dur} dias após a OS ({dur / prazo_c:.2f}× o prazo original)")
            v2.metric("Previsão (mediana)", fmt_data(soma_dias(data_os, q["p50"])),
                       help=AJUDA["mediana"] + "\n\n" + AJUDA["epoca"])
            v2.caption(f"a obra terminou {n_dias(dur - q['p50'])} {'depois' if dur > q['p50'] else 'antes'}")
            v3.metric("Previsão (pessimista)", fmt_data(soma_dias(data_os, q["p80"])), help=AJUDA["pessimista"])
            v3.caption(f"a obra terminou {n_dias(dur - q['p80'])} {'depois' if dur > q['p80'] else 'antes'}")
            posicao = (f"{n_dias(atraso)} {'depois' if atraso > 0 else 'antes'} do prazo original"
                       if atraso != 0 else "exatamente no prazo original")
            st.markdown(f"A obra terminou **{posicao}**, na faixa **{nome_faixa}**.", help=AJUDA["faixa"])
            if adit_c:
                st.caption(
                    f"Os +{adit_c} dias de aditivo não entram nesta comparação: vale o prazo original.",
                    help="O CSV não traz a data de assinatura de cada aditivo, então não dá para saber qual era o "
                         "prazo aditivado em cada dia da obra.",
                )
        else:
            decorridos_c = max((pd.Timestamp(data_corte) - pd.Timestamp(data_os)).days, 0)
            st.divider()
            st.metric("Dias decorridos", fmt_dias(decorridos_c))
            st.caption(f"{100 * decorridos_c / prazo_c:.0f}% do prazo original, em {fmt_data(data_corte)}")
            if grupo == "SUSPENSO_PARALISADO":
                st.warning("Contrato suspenso/paralisado: fica fora do ajuste do modelo. A previsão abaixo é só ilustrativa.")
            painel_previsao(modelo, data_os, prazo_c, vigente_c if adit_c else None, decorridos_c)

# --------------------------------------------------------------------------- #
# Aba 3 — Contratos em andamento
# --------------------------------------------------------------------------- #
with aba_and:
    st.subheader(
        "Risco dos contratos em andamento",
        help="Para cada obra em andamento: dado que ela **ainda não terminou**, qual a chance de terminar até o prazo "
             "original e até 30, 60, 120, 150 e 180 dias depois dele. " + AJUDA["probabilidade"],
    )
    if em_andamento.empty:
        st.info("Nenhum contrato em andamento na base.")
    else:
        cols = {
            "id_contrato": "Contrato",
            "Numero_contrato": "Nº contrato",
            "Nome_predio": "Prédio",
            "prazo_original_dias": "Prazo original (d)",
            "dias_aditivados_app": "Aditivos (d)",
            "prazo_vigente_atual_dias": "Prazo vigente (d)",
            "tempo_observado_dias": "Decorridos (d)",
        }
        rotulo_marco = {m: ("Até o prazo" if m == 0 else f"Até +{m} dias") for m in MARCOS}
        for m in MARCOS:
            cols[f"prob_ate_original_mais_{m}"] = rotulo_marco[m]
        cols.update({
            "prob_estourar_prazo_vigente": "Prob. de passar do vigente",
            "termino_mediana": "Término (mediana)",
            "termino_p80": "Término (pessimista)",
        })
        tabela = em_andamento[[k for k in cols if k in em_andamento.columns]].rename(columns=cols)
        for k in ("Término (mediana)", "Término (pessimista)"):
            tabela[k] = pd.to_datetime(tabela[k]).dt.date

        barra = lambda ajuda: st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent", help=ajuda)
        config = {
            "Prob. de passar do vigente": barra("Chance de a obra terminar depois do prazo vigente (original + aditivos)."),
            "Término (mediana)": st.column_config.DateColumn(format="DD/MM/YYYY", help=AJUDA["mediana"]),
            "Término (pessimista)": st.column_config.DateColumn(format="DD/MM/YYYY", help=AJUDA["pessimista"]),
        }
        config.update({rotulo_marco[m]: barra(AJUDA["probabilidade"]) for m in MARCOS})
        st.dataframe(tabela, hide_index=True, width="stretch", column_config=config)
        st.download_button(
            "Baixar CSV",
            tabela.to_csv(index=False).encode("utf-8-sig"),
            file_name="risco_contratos_em_andamento.csv",
            mime="text/csv",
        )
