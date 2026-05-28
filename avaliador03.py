import logging
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIGURACOES
# =========================================================

ARQUIVO_CSV = "semRAG4.1mini.csv"

# Coluna com o rotulo real:
# 0 = verdadeiro
# 1 = falso
COLUNA_LABEL_REAL = "label_manual"

# Coluna que contem o SCORE da comparacao.
# Troque para o nome da coluna real do seu CSV.
# Exemplos possiveis:
# "final_score", "similaridade", "semantic_score", "score"
COLUNA_SCORE = "final_score"

# Colunas de texto usadas para comparar respostas
COLUNA_RESPOSTA_FAQ = "Resposta FAQ"
COLUNA_RESPOSTA_CHATBOT = "Resposta sem RAG"

# Colunas de score (quando ja existem no CSV)
COLUNA_SEMANTIC_SCORE = "semantic_score"
COLUNA_LEXICAL_SCORE = "lexical_score"

# Thresholds calculados automaticamente
LISTA_THRESHOLDS = [round(x, 1) for x in np.arange(0.5, 1.0, 0.1)]

# Como voce definiu:
LABEL_VERDADEIRO = 0
LABEL_FALSO = 1

# Para avaliador de FAQ, normalmente:
# score alto = resposta mais parecida com FAQ = verdadeiro
SCORE_ALTO_INDICA_VERDADEIRO = True

# Metrica usada para escolher o melhor threshold
METRICA_MELHOR = "f1_verdadeiro"


# =========================================================
# FUNCOES AUXILIARES
# =========================================================

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import logging as transformers_logging
        except ImportError as exc:
            raise ImportError(
                "Faltam dependencias para calcular similaridade semantica. "
                "Instale 'sentence-transformers' e 'transformers'."
            ) from exc

        transformers_logging.set_verbosity_error()
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        _MODEL = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _MODEL


def semantic_similarity(text1: str, text2: str) -> float:
    emb = _get_model().encode([text1, text2])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(score)


def lexical_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def converter_para_float(valor):
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")

    try:
        return float(valor)
    except ValueError:
        return np.nan


def gerar_label_pred(score, threshold):
    """
    Gera a predicao a partir do score e do threshold.

    Se SCORE_ALTO_INDICA_VERDADEIRO = True:
        score >= threshold -> 0 (verdadeiro)
        score < threshold  -> 1 (falso)
    """
    if pd.isna(score):
        return np.nan

    if SCORE_ALTO_INDICA_VERDADEIRO:
        return LABEL_VERDADEIRO if score >= threshold else LABEL_FALSO

    return LABEL_FALSO if score >= threshold else LABEL_VERDADEIRO


def preparar_dados(df):
    dados = df.copy()

    if COLUNA_LABEL_REAL not in dados.columns:
        raise ValueError(
            f"A coluna '{COLUNA_LABEL_REAL}' nao existe no CSV.\n"
            f"Colunas disponiveis: {dados.columns.tolist()}"
        )

    dados = adicionar_final_score(dados)

    if COLUNA_SCORE not in dados.columns:
        raise ValueError(
            f"A coluna de score '{COLUNA_SCORE}' nao existe no CSV.\n"
            f"Colunas disponiveis: {dados.columns.tolist()}"
        )

    dados[COLUNA_LABEL_REAL] = pd.to_numeric(
        dados[COLUNA_LABEL_REAL],
        errors="coerce"
    )

    dados[COLUNA_SCORE] = dados[COLUNA_SCORE].apply(converter_para_float)

    dados = dados.dropna(subset=[COLUNA_LABEL_REAL, COLUNA_SCORE])
    dados[COLUNA_LABEL_REAL] = dados[COLUNA_LABEL_REAL].astype(int)

    return dados


def adicionar_final_score(dados):
    if COLUNA_SCORE not in dados.columns:
        dados[COLUNA_SCORE] = np.nan

    if (
        COLUNA_SEMANTIC_SCORE in dados.columns
        and COLUNA_LEXICAL_SCORE in dados.columns
    ):
        dados[COLUNA_SEMANTIC_SCORE] = dados[COLUNA_SEMANTIC_SCORE].apply(
            converter_para_float
        )
        dados[COLUNA_LEXICAL_SCORE] = dados[COLUNA_LEXICAL_SCORE].apply(
            converter_para_float
        )

        mask = (
            dados[COLUNA_SCORE].isna()
            & dados[COLUNA_SEMANTIC_SCORE].notna()
            & dados[COLUNA_LEXICAL_SCORE].notna()
        )

        dados.loc[mask, COLUNA_SCORE] = (
            0.8 * dados.loc[mask, COLUNA_SEMANTIC_SCORE]
            + 0.2 * dados.loc[mask, COLUNA_LEXICAL_SCORE]
        )

    missing_mask = dados[COLUNA_SCORE].isna()

    if missing_mask.any():
        if (
            COLUNA_RESPOSTA_FAQ not in dados.columns
            or COLUNA_RESPOSTA_CHATBOT not in dados.columns
        ):
            raise ValueError(
                "Nao foi possivel calcular final_score. "
                "Inclua as colunas de texto ou os scores no CSV.\n"
                f"Esperado: '{COLUNA_RESPOSTA_FAQ}', "
                f"'{COLUNA_RESPOSTA_CHATBOT}' ou "
                f"'{COLUNA_SEMANTIC_SCORE}'/'{COLUNA_LEXICAL_SCORE}'."
            )

        if COLUNA_SEMANTIC_SCORE not in dados.columns:
            dados[COLUNA_SEMANTIC_SCORE] = np.nan
        if COLUNA_LEXICAL_SCORE not in dados.columns:
            dados[COLUNA_LEXICAL_SCORE] = np.nan

        for idx in dados[missing_mask].index:
            faq = dados.at[idx, COLUNA_RESPOSTA_FAQ]
            chatbot = dados.at[idx, COLUNA_RESPOSTA_CHATBOT]

            if pd.isna(faq) or pd.isna(chatbot):
                continue

            sem = semantic_similarity(str(faq), str(chatbot))
            lex = lexical_similarity(str(faq), str(chatbot))
            final = 0.8 * sem + 0.2 * lex

            dados.at[idx, COLUNA_SEMANTIC_SCORE] = round(sem, 4)
            dados.at[idx, COLUNA_LEXICAL_SCORE] = round(lex, 4)
            dados.at[idx, COLUNA_SCORE] = round(final, 4)

    return dados


def calcular_metricas_threshold(dados, threshold):
    dados_temp = dados.copy()

    dados_temp["label_pred"] = dados_temp[COLUNA_SCORE].apply(
        lambda score: gerar_label_pred(score, threshold)
    )

    dados_temp = dados_temp.dropna(subset=["label_pred"])
    dados_temp["label_pred"] = dados_temp["label_pred"].astype(int)

    y_true = dados_temp[COLUNA_LABEL_REAL]
    y_pred = dados_temp["label_pred"]

    metricas = {
        "threshold": threshold,

        "accuracy": accuracy_score(y_true, y_pred),

        "precision_verdadeiro": precision_score(
            y_true,
            y_pred,
            pos_label=LABEL_VERDADEIRO,
            zero_division=0
        ),
        "recall_verdadeiro": recall_score(
            y_true,
            y_pred,
            pos_label=LABEL_VERDADEIRO,
            zero_division=0
        ),
        "f1_verdadeiro": f1_score(
            y_true,
            y_pred,
            pos_label=LABEL_VERDADEIRO,
            zero_division=0
        ),

        "precision_falso": precision_score(
            y_true,
            y_pred,
            pos_label=LABEL_FALSO,
            zero_division=0
        ),
        "recall_falso": recall_score(
            y_true,
            y_pred,
            pos_label=LABEL_FALSO,
            zero_division=0
        ),
        "f1_falso": f1_score(
            y_true,
            y_pred,
            pos_label=LABEL_FALSO,
            zero_division=0
        ),

        "qtd_linhas_usadas": len(dados_temp)
    }

    dados_temp["threshold_avaliado"] = threshold

    return metricas, dados_temp


def avaliar_todos_thresholds(dados):
    resultados_metricas = []
    resultados_predicoes = []

    for threshold in LISTA_THRESHOLDS:
        metricas, dados_preditos = calcular_metricas_threshold(
            dados=dados,
            threshold=threshold
        )

        resultados_metricas.append(metricas)
        resultados_predicoes.append(dados_preditos)

    df_metricas = pd.DataFrame(resultados_metricas)
    df_predicoes = pd.concat(resultados_predicoes, ignore_index=True)

    return df_metricas, df_predicoes


# =========================================================
# EXECUCAO
# =========================================================

if __name__ == "__main__":
    df = pd.read_csv(ARQUIVO_CSV)

    print("\nColunas encontradas no CSV:")
    print(df.columns.tolist())

    dados = preparar_dados(df)

    df_metricas, df_predicoes = avaliar_todos_thresholds(dados)

    print("\n================ METRICAS POR THRESHOLD ================")
    print(df_metricas)

    melhor_threshold = df_metricas.sort_values(
        by=METRICA_MELHOR,
        ascending=False
    ).iloc[0]

    print("\n================ MELHOR THRESHOLD ================")
    print(melhor_threshold)

    df_metricas.to_csv(
        "metricas_por_threshold.csv",
        index=False,
        sep=";",
        decimal="," 
    )

    df_predicoes.to_csv(
        "predicoes_por_threshold.csv",
        index=False,
        sep=";",
        decimal="," 
    )

    print("\nArquivos gerados:")
    print("- metricas_por_threshold.csv")
    print("- predicoes_por_threshold.csv")