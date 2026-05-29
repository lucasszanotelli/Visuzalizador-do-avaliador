import csv
import argparse
import logging
from pathlib import Path
from py_compile import main

import pandas as pd
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging


_MODEL = None

COL_PERGUNTA = "Pergunta"
COL_FAQ = "Resposta FAQ"
COL_RAG = "Resposta sem RAG"

COL_SEMANTIC = "semantic_score"
COL_LEXICAL = "lexical_score"
COL_FINAL = "final_score"
COL_CLASSIFICACAO = "classificacao"


def _get_model() -> SentenceTransformer:
    global _MODEL

    if _MODEL is None:
        transformers_logging.set_verbosity_error()
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        _MODEL = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    return _MODEL


def carregar_csv(caminho_csv: str) -> pd.DataFrame:
    """
    Lê CSV tentando detectar separador automaticamente.
    Funciona bem com CSV exportado de Excel/Google Sheets.
    """

    encodings = ["utf-8-sig", "utf-8", "latin1"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                caminho_csv,
                sep=None,
                engine="python",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("Não foi possível ler o CSV. Verifique a codificação do arquivo.")


def validar_colunas(df: pd.DataFrame):
    colunas_obrigatorias = [COL_PERGUNTA, COL_FAQ, COL_RAG]

    colunas_faltando = [
        coluna for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_faltando:
        raise ValueError(
            "O CSV não possui as colunas obrigatórias: "
            + ", ".join(colunas_faltando)
        )


def lexical_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(
        None,
        str(text1).lower(),
        str(text2).lower()
    ).ratio()


def classificar_score(final_score: float) -> str:
    if final_score >= 0.85:
        return "excelente"
    elif final_score >= 0.70:
        return "aceitável"
    elif final_score >= 0.50:
        return "fraco"
    else:
        return "ruim"


def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    validar_colunas(df)

    faq_texts = df[COL_FAQ].fillna("").astype(str).tolist()
    rag_texts = df[COL_RAG].fillna("").astype(str).tolist()

    model = _get_model()

    faq_embeddings = model.encode(
        faq_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    rag_embeddings = model.encode(
        rag_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    semantic_scores = []

    for emb_faq, emb_rag in zip(faq_embeddings, rag_embeddings):
        semantic_score = float((emb_faq * emb_rag).sum())
        semantic_scores.append(semantic_score)

    lexical_scores = [
        lexical_similarity(faq, rag)
        for faq, rag in zip(faq_texts, rag_texts)
    ]

    final_scores = [
        0.8 * semantic + 0.2 * lexical
        for semantic, lexical in zip(semantic_scores, lexical_scores)
    ]

    df[COL_SEMANTIC] = [round(score, 4) for score in semantic_scores]
    df[COL_LEXICAL] = [round(score, 4) for score in lexical_scores]
    df[COL_FINAL] = [round(score, 4) for score in final_scores]
    df[COL_CLASSIFICACAO] = [
        classificar_score(score)
        for score in final_scores
    ]

    return df


def salvar_csv(df: pd.DataFrame, caminho_saida: str):
    """
    Salva com:
    - sep=';' para evitar conflito com decimal brasileiro
    - decimal=',' para sair como 0,7494
    - quoting=QUOTE_ALL para não quebrar textos grandes com vírgula ou quebra de linha
    """

    df.to_csv(
        caminho_saida,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )


def salvar_html_tabela(df: pd.DataFrame, caminho_saida: str):
        df_html = df.copy()

        for coluna in [COL_SEMANTIC, COL_LEXICAL, COL_FINAL]:
                if coluna in df_html.columns:
                        df_html[coluna] = df_html[coluna].apply(
                                lambda valor: (
                                        f"{valor:.4f}".replace(".", ",")
                                        if isinstance(valor, (int, float))
                                        else valor
                                )
                        )

        html_table = df_html.to_html(
                index=False,
                classes="metrics-table",
                border=0,
        )

        html = f"""<!doctype html>
<html lang=\"pt-BR\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Metricas por pergunta</title>
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
        <link
            href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap\"
            rel=\"stylesheet\"
        />
        <style>
            :root {{
                --bg-1: #0f172a;
                --bg-2: #1e293b;
                --card: #0b1324;
                --text: #e2e8f0;
                --muted: #94a3b8;
                --accent: #ffb703;
                --line: rgba(148, 163, 184, 0.2);
                --shadow: 0 20px 50px rgba(15, 23, 42, 0.45);
                --radius: 16px;
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                font-family: "Space Grotesk", system-ui, -apple-system, sans-serif;
                background: radial-gradient(circle at top left, #1f2b58, transparent 55%),
                    radial-gradient(circle at 80% 10%, #37212b, transparent 40%),
                    linear-gradient(160deg, var(--bg-1), var(--bg-2));
                color: var(--text);
                min-height: 100vh;
            }}

            .page {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 48px 24px 72px;
            }}

            header {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 20px;
                align-items: end;
                margin-bottom: 24px;
            }}

            h1 {{
                font-family: "Fraunces", "Space Grotesk", serif;
                font-size: clamp(2rem, 4vw, 3rem);
                margin: 0 0 8px;
            }}

            .subtitle {{
                margin: 0;
                color: var(--muted);
                font-size: 1rem;
            }}

            .card {{
                background: var(--card);
                border-radius: var(--radius);
                padding: 18px 20px;
                box-shadow: var(--shadow);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}

            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.95rem;
            }}

            .metrics-table th,
            .metrics-table td {{
                padding: 12px 14px;
                border-bottom: 1px solid var(--line);
                text-align: left;
                vertical-align: top;
                white-space: pre-wrap;
            }}

            .metrics-table th {{
                color: var(--muted);
                font-weight: 600;
            }}

            .metrics-table tbody tr:hover {{
                background: rgba(255, 183, 3, 0.12);
            }}

            .footer {{
                margin-top: 18px;
                color: var(--muted);
                font-size: 0.85rem;
            }}

            @media (max-width: 720px) {{
                .metrics-table th,
                .metrics-table td {{
                    padding: 10px 12px;
                    font-size: 0.85rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class=\"page\">
            <header>
                <div>
                    <h1>Metricas por pergunta</h1>
                    <p class=\"subtitle\">Tabela gerada pelo avaliador 2.</p>
                </div>
                <div class=\"card\">
                    <strong>Total de linhas:</strong> {len(df_html)}
                </div>
            </header>

            <section class=\"card\">
                {html_table}
            </section>

            <div class=\"footer\">
                Arquivo gerado automaticamente pelo avaliador 2.
            </div>
        </div>
    </body>
</html>
"""

        Path(caminho_saida).write_text(html, encoding="utf-8")


def main():
    caminho_entrada = Path("semRAG4.1mini.csv")
    caminho_saida = Path("metricas_por_pergunta.csv")
    caminho_saida_html = Path("metricas_por_pergunta.html")

    if not caminho_entrada.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_entrada}")

    df = carregar_csv(str(caminho_entrada))
    df_resultado = calcular_metricas(df)
    salvar_csv(df_resultado, str(caminho_saida))
    salvar_html_tabela(df_resultado, str(caminho_saida_html))

    print(f"Arquivo gerado com sucesso: {caminho_saida}")
    print(f"Tabela HTML gerada com sucesso: {caminho_saida_html}")


if __name__ == "__main__":
    main()