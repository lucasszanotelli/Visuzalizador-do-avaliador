from __future__ import annotations

from pathlib import Path
import html

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_DIR = ROOT_DIR / "comparacao-respostas"
OUTPUT_HTML = SCRIPT_DIR / "comparacao_respostas.html"

COL_PERGUNTA = "Pergunta"
COL_FAQ = "Resposta FAQ"
COL_RESPOSTA_IA = "Resposta IA"
COL_SEMANTIC = "semantic_score"
COL_LEXICAL = "lexical_score"
COL_FINAL = "final_score"
COL_CLASSIFICACAO = "classificacao"

SCORE_COLUNAS = [COL_SEMANTIC, COL_LEXICAL, COL_FINAL]


def carregar_csv(caminho_csv: Path) -> pd.DataFrame:
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

    raise ValueError(
        f"Nao foi possivel ler o arquivo {caminho_csv.name}. Verifique a codificacao."
    )


def identificar_coluna_resposta(df: pd.DataFrame) -> str:
    for coluna in ("Resposta com RAG", "Resposta sem RAG"):
        if coluna in df.columns:
            return coluna

    candidatos = [
        coluna
        for coluna in df.columns
        if coluna.startswith("Resposta") and coluna != COL_FAQ
    ]

    if candidatos:
        return candidatos[0]

    raise ValueError(
        f"Nao encontrei uma coluna de resposta IA. Colunas disponiveis: {df.columns.tolist()}"
    )


def rotulo_origem(caminho_csv: Path) -> str:
    nome = caminho_csv.stem.upper()

    if "4.1-MINI" in nome:
        modelo = "ChatGPT 4.1-MINI"
    elif "5.3" in nome:
        modelo = "ChatGPT 5.3"
    else:
        modelo = caminho_csv.stem

    if "COM RAG" in nome:
        sufixo = "com RAG"
    elif "SEM RAG" in nome:
        sufixo = "sem RAG"
    else:
        sufixo = "comparacao"

    return f"{modelo} - {sufixo}"


def converter_score(valor: object) -> float | None:
    if valor is None:
        return None

    texto = str(valor).strip()
    if not texto:
        return None

    texto = texto.replace(",", ".")

    try:
        return float(texto)
    except ValueError:
        return None


def formatar_score(valor: object) -> str:
    score = converter_score(valor)
    if score is None:
        return ""
    return f"{score:.4f}".replace(".", ",")


def preparar_dataframe(df: pd.DataFrame, origem: str) -> pd.DataFrame:
    dados = df.copy()
    resposta_coluna = identificar_coluna_resposta(dados)

    dados = dados.rename(columns={resposta_coluna: COL_RESPOSTA_IA})
    dados.insert(0, "Origem", origem)

    for coluna in SCORE_COLUNAS:
        if coluna in dados.columns:
            dados[coluna] = dados[coluna].apply(formatar_score)

    return dados


def resumo_por_origem(df: pd.DataFrame) -> dict[str, object]:
    final_scores = df[COL_FINAL].apply(converter_score) if COL_FINAL in df.columns else pd.Series(dtype=float)
    semantic_scores = df[COL_SEMANTIC].apply(converter_score) if COL_SEMANTIC in df.columns else pd.Series(dtype=float)
    lexical_scores = df[COL_LEXICAL].apply(converter_score) if COL_LEXICAL in df.columns else pd.Series(dtype=float)

    classificacoes = df[COL_CLASSIFICACAO].value_counts(dropna=False).to_dict() if COL_CLASSIFICACAO in df.columns else {}

    return {
        "linhas": int(len(df)),
        "media_final": final_scores.mean() if not final_scores.empty else None,
        "media_semantica": semantic_scores.mean() if not semantic_scores.empty else None,
        "media_lexical": lexical_scores.mean() if not lexical_scores.empty else None,
        "excelente": int(classificacoes.get("excelente", 0)),
        "aceitavel": int(classificacoes.get("aceitável", 0)),
        "fraco": int(classificacoes.get("fraco", 0)),
        "ruim": int(classificacoes.get("ruim", 0)),
    }


def construir_comparacao_por_pergunta(dados_por_origem: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    linhas = []

    base_combinada = pd.concat(
        [
            df[[COL_PERGUNTA, COL_FAQ, COL_FINAL, COL_CLASSIFICACAO]].copy().assign(Origem=origem)
            for origem, df in dados_por_origem
        ],
        ignore_index=True,
    )

    for (pergunta, resposta_faq), grupo in base_combinada.groupby([COL_PERGUNTA, COL_FAQ], sort=False):
        linha = {
            COL_PERGUNTA: pergunta,
            COL_FAQ: resposta_faq,
        }

        melhor_origem = None
        melhor_score = None

        for _, registro in grupo.iterrows():
            origem = registro["Origem"]
            score = converter_score(registro[COL_FINAL])
            classificacao = registro[COL_CLASSIFICACAO]

            linha[f"{origem} - final_score"] = formatar_score(score)
            linha[f"{origem} - classificacao"] = classificacao

            if score is not None and (melhor_score is None or score > melhor_score):
                melhor_score = score
                melhor_origem = origem

        linha["Melhor IA"] = melhor_origem or ""
        linha["Melhor final_score"] = formatar_score(melhor_score)
        linhas.append(linha)

    df_comparacao = pd.DataFrame(linhas)

    colunas_fixas = [COL_PERGUNTA, COL_FAQ, "Melhor IA", "Melhor final_score"]
    colunas_origem = []

    for origem, _ in dados_por_origem:
        colunas_origem.extend([f"{origem} - final_score", f"{origem} - classificacao"])

    colunas = [coluna for coluna in colunas_fixas + colunas_origem if coluna in df_comparacao.columns]
    return df_comparacao[colunas]


def dataframe_para_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="metrics-table", border=0, escape=True)


def linha_resumo_html(origem: str, resumo: dict[str, object]) -> str:
    return f"""
    <article class="summary-card">
        <p class="summary-label">{html.escape(origem)}</p>
        <h3>{resumo['linhas']} linhas</h3>
        <div class="summary-grid">
            <div><span>Média final</span><strong>{formatar_score(resumo['media_final'])}</strong></div>
            <div><span>Média semântica</span><strong>{formatar_score(resumo['media_semantica'])}</strong></div>
            <div><span>Média lexical</span><strong>{formatar_score(resumo['media_lexical'])}</strong></div>
            <div><span>Excelente</span><strong>{resumo['excelente']}</strong></div>
            <div><span>Aceitável</span><strong>{resumo['aceitavel']}</strong></div>
            <div><span>Fraco</span><strong>{resumo['fraco']}</strong></div>
            <div><span>Ruim</span><strong>{resumo['ruim']}</strong></div>
        </div>
    </article>
    """


def construir_html(dados_por_origem: list[tuple[str, pd.DataFrame]]) -> str:
    totais_linhas = sum(len(df) for _, df in dados_por_origem)
    comparacao_por_pergunta = construir_comparacao_por_pergunta(dados_por_origem)

    resumo_html = "".join(
        linha_resumo_html(origem, resumo_por_origem(df))
        for origem, df in dados_por_origem
    )

    comparacao_html = dataframe_para_html(comparacao_por_pergunta)

    secoes_detalhe = []
    for origem, df in dados_por_origem:
        secoes_detalhe.append(
            f"""
            <details class="source-details">
                <summary>{html.escape(origem)} - {len(df)} linhas</summary>
                <div class="details-body">
                    {dataframe_para_html(df)}
                </div>
            </details>
            """
        )

    detalhes_html = "".join(secoes_detalhe)

    return f"""<!doctype html>
<html lang="pt-BR">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Comparação de respostas por IA</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />
        <style>
            :root {{
                --bg: #08111f;
                --bg-soft: #101a31;
                --panel: rgba(11, 18, 32, 0.88);
                --text: #e5eefc;
                --muted: #94a3b8;
                --accent: #f7b267;
                --accent-2: #74c69d;
                --line: rgba(148, 163, 184, 0.18);
                --shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
                --radius: 20px;
            }}

            * {{ box-sizing: border-box; }}

            body {{
                margin: 0;
                font-family: "Space Grotesk", system-ui, sans-serif;
                color: var(--text);
                background:
                    radial-gradient(circle at 12% 10%, rgba(116, 198, 157, 0.18), transparent 28%),
                    radial-gradient(circle at 88% 0%, rgba(247, 178, 103, 0.20), transparent 25%),
                    linear-gradient(160deg, var(--bg), var(--bg-soft));
                min-height: 100vh;
            }}

            .page {{
                width: min(1500px, calc(100% - 32px));
                margin: 0 auto;
                padding: 32px 0 56px;
            }}

            .hero {{
                background: linear-gradient(135deg, rgba(12, 22, 39, 0.95), rgba(16, 26, 49, 0.88));
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 28px;
                padding: 32px;
                box-shadow: var(--shadow);
                overflow: hidden;
                position: relative;
            }}

            .hero::after {{
                content: "";
                position: absolute;
                inset: auto -120px -140px auto;
                width: 320px;
                height: 320px;
                background: radial-gradient(circle, rgba(247, 178, 103, 0.26), transparent 65%);
                pointer-events: none;
            }}

            .eyebrow {{
                display: inline-flex;
                gap: 10px;
                align-items: center;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(116, 198, 157, 0.12);
                color: #b9f6d3;
                border: 1px solid rgba(116, 198, 157, 0.18);
                font-size: 0.9rem;
                letter-spacing: 0.02em;
            }}

            h1 {{
                font-family: "Fraunces", Georgia, serif;
                font-size: clamp(2.3rem, 4vw, 4rem);
                line-height: 1.02;
                margin: 18px 0 12px;
                max-width: 10ch;
            }}

            .subtitle {{
                margin: 0;
                max-width: 70ch;
                color: var(--muted);
                font-size: 1.03rem;
                line-height: 1.6;
            }}

            .hero-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
                gap: 14px;
                margin-top: 24px;
            }}

            .stat {{
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 16px 18px;
            }}

            .stat span {{ display: block; color: var(--muted); font-size: 0.88rem; margin-bottom: 8px; }}
            .stat strong {{ font-size: 1.15rem; }}

            .section {{ margin-top: 26px; }}

            .section-title {{
                display: flex;
                justify-content: space-between;
                align-items: end;
                gap: 16px;
                margin: 0 0 14px;
            }}

            .section-title h2 {{
                margin: 0;
                font-size: 1.35rem;
            }}

            .section-title p {{
                margin: 0;
                color: var(--muted);
                font-size: 0.95rem;
            }}

            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
            }}

            .summary-card, .panel {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }}

            .summary-card {{ padding: 18px 18px 16px; }}

            .summary-label {{
                margin: 0;
                color: var(--accent);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.76rem;
                font-weight: 700;
            }}

            .summary-card h3 {{ margin: 10px 0 16px; font-size: 1.35rem; }}

            .summary-card .summary-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
            }}

            .summary-card .summary-grid span {{
                display: block;
                color: var(--muted);
                font-size: 0.8rem;
                margin-bottom: 4px;
            }}

            .summary-card .summary-grid strong {{ font-size: 1rem; }}

            .panel {{ padding: 18px; }}

            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.92rem;
            }}

            .metrics-table th,
            .metrics-table td {{
                border-bottom: 1px solid var(--line);
                padding: 12px 14px;
                text-align: left;
                vertical-align: top;
                white-space: pre-wrap;
            }}

            .metrics-table th {{
                color: #cbd5e1;
                font-weight: 600;
                position: sticky;
                top: 0;
                background: linear-gradient(180deg, rgba(12, 22, 39, 0.98), rgba(12, 22, 39, 0.92));
                backdrop-filter: blur(12px);
                z-index: 1;
            }}

            .metrics-table tbody tr:hover {{ background: rgba(247, 178, 103, 0.09); }}

            .table-wrap {{
                max-height: 760px;
                overflow: auto;
                border-radius: 16px;
                border: 1px solid var(--line);
            }}

            .source-details {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                overflow: hidden;
            }}

            .source-details + .source-details {{ margin-top: 14px; }}

            .source-details summary {{
                cursor: pointer;
                list-style: none;
                padding: 18px 20px;
                font-weight: 700;
                font-size: 1.02rem;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), transparent);
            }}

            .source-details summary::-webkit-details-marker {{ display: none; }}

            .details-body {{ padding: 0 18px 18px; }}

            .footer {{
                margin-top: 18px;
                color: var(--muted);
                font-size: 0.85rem;
                text-align: right;
            }}

            @media (max-width: 900px) {{
                .page {{ width: min(100% - 18px, 1500px); padding-top: 18px; }}
                .hero {{ padding: 22px; border-radius: 22px; }}
                .summary-card .summary-grid {{ grid-template-columns: 1fr; }}
                .metrics-table th,
                .metrics-table td {{ padding: 10px 12px; font-size: 0.84rem; }}
            }}
        </style>
    </head>
    <body>
        <main class="page">
            <section class="hero">
                <div class="eyebrow">Comparação consolidada de respostas de IA</div>
                <h1>Um único HTML para comparar as avaliações</h1>
                <p class="subtitle">
                    Este relatório consolida os CSVs de <strong>{html.escape(INPUT_DIR.name)}</strong>,
                    exibindo um resumo por IA, a comparação por pergunta e os dados completos de cada arquivo.
                </p>
                <div class="hero-stats">
                    <div class="stat"><span>Arquivos analisados</span><strong>{len(dados_por_origem)}</strong></div>
                    <div class="stat"><span>Linhas totais</span><strong>{totais_linhas}</strong></div>
                    <div class="stat"><span>Perguntas comparadas</span><strong>{len(comparacao_por_pergunta)}</strong></div>
                </div>
            </section>

            <section class="section">
                <div class="section-title">
                    <div>
                        <h2>Resumo por IA</h2>
                        <p>Média dos scores e distribuição das classificações em cada CSV.</p>
                    </div>
                </div>
                <div class="summary-grid">
                    {resumo_html}
                </div>
            </section>

            <section class="section">
                <div class="section-title">
                    <div>
                        <h2>Comparação por pergunta</h2>
                        <p>As notas finais são exibidas lado a lado para facilitar a leitura da melhor resposta em cada questão.</p>
                    </div>
                </div>
                <div class="panel">
                    <div class="table-wrap">
                        {comparacao_html}
                    </div>
                </div>
            </section>

            <section class="section">
                <div class="section-title">
                    <div>
                        <h2>Detalhe por arquivo</h2>
                        <p>Expanda cada bloco para ver os dados completos de origem sem perder a consolidação geral.</p>
                    </div>
                </div>
                {detalhes_html}
            </section>

            <div class="footer">Arquivo gerado automaticamente a partir da pasta comparacao-respostas.</div>
        </main>
    </body>
</html>
"""


def carregar_dados() -> list[tuple[str, pd.DataFrame]]:
    arquivos = sorted(INPUT_DIR.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {INPUT_DIR}.")

    dados_por_origem: list[tuple[str, pd.DataFrame]] = []

    for caminho in arquivos:
        origem = rotulo_origem(caminho)
        df = carregar_csv(caminho)
        dados_por_origem.append((origem, preparar_dataframe(df, origem)))

    return dados_por_origem


def main() -> None:
    dados_por_origem = carregar_dados()
    html_resultado = construir_html(dados_por_origem)
    OUTPUT_HTML.write_text(html_resultado, encoding="utf-8")

    print(f"HTML gerado com sucesso: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()