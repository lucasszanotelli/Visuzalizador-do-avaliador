from __future__ import annotations

from pathlib import Path
import html

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_DIR = ROOT_DIR / "metricas"
OUTPUT_HTML = SCRIPT_DIR / "comparacao_metricas_avaliadores.html"

METRICAS = [
    "accuracy",
    "precision_verdadeiro",
    "recall_verdadeiro",
    "f1_verdadeiro",
    "precision_falso",
    "recall_falso",
    "f1_falso",
]


def carregar_csv(caminho_csv: Path) -> pd.DataFrame:
    return pd.read_csv(caminho_csv, sep=";", decimal=",")


def source_label(caminho_csv: Path) -> str:
    stem = caminho_csv.stem
    label = stem.replace("metricas_por_threshold", "").strip(" _-.")
    return label or stem


def format_score(valor: object, digits: int = 4) -> str:
    if valor is None or pd.isna(valor):
        return ""
    return f"{float(valor):.{digits}f}".replace(".", ",")


def format_threshold(valor: object) -> str:
    if valor is None or pd.isna(valor):
        return ""
    return f"{float(valor):.1f}".replace(".", ",")


def melhor_threshold(df: pd.DataFrame) -> pd.Series | None:
    if "f1_verdadeiro" not in df.columns:
        return None
    df_valid = df.dropna(subset=["f1_verdadeiro"])
    if df_valid.empty:
        return None
    return df_valid.sort_values(by="f1_verdadeiro", ascending=False).iloc[0]


def resumo_por_origem(label: str, df: pd.DataFrame) -> str:
    row = melhor_threshold(df)
    linhas = int(df["qtd_linhas_usadas"].iloc[0]) if "qtd_linhas_usadas" in df.columns else len(df)

    if row is None:
        return f"""
        <article class="summary-card">
            <p class="summary-label">{html.escape(label)}</p>
            <h3>{linhas} linhas</h3>
            <p class="summary-note">Sem f1_verdadeiro disponivel.</p>
        </article>
        """

    return f"""
    <article class="summary-card">
        <p class="summary-label">{html.escape(label)}</p>
        <h3>{linhas} linhas</h3>
        <div class="summary-grid">
            <div><span>Melhor threshold</span><strong>{format_threshold(row['threshold'])}</strong></div>
            <div><span>F1 verdadeiro</span><strong>{format_score(row['f1_verdadeiro'])}</strong></div>
            <div><span>Accuracy</span><strong>{format_score(row['accuracy'])}</strong></div>
            <div><span>Recall verdadeiro</span><strong>{format_score(row['recall_verdadeiro'])}</strong></div>
        </div>
    </article>
    """


def tabela_metricas_por_origem(dados: list[tuple[str, pd.DataFrame]], metrica: str) -> str:
    thresholds = sorted({
        round(float(valor), 2)
        for _, df in dados
        for valor in df["threshold"].dropna().tolist()
    })

    tabela = pd.DataFrame({"threshold": [format_threshold(valor) for valor in thresholds]})

    for label, df in dados:
        df = df.copy()
        df["threshold"] = df["threshold"].apply(lambda valor: round(float(valor), 2))
        valores = df.set_index("threshold")[metrica].to_dict()
        tabela[label] = [format_score(valores.get(threshold)) for threshold in thresholds]

    return tabela.to_html(index=False, classes="metrics-table", border=0, escape=True)


def detalhes_por_origem(dados: list[tuple[str, pd.DataFrame]]) -> str:
    secoes = []
    for label, df in dados:
        df_formatado = df.copy()
        df_formatado["threshold"] = df_formatado["threshold"].apply(format_threshold)
        for coluna in METRICAS:
            if coluna in df_formatado.columns:
                df_formatado[coluna] = df_formatado[coluna].apply(format_score)

        tabela_html = df_formatado.to_html(index=False, classes="metrics-table", border=0, escape=True)

        secoes.append(
            f"""
            <details class="source-details">
                <summary>{html.escape(label)}</summary>
                <div class="details-body">
                    {tabela_html}
                </div>
            </details>
            """
        )

    return "".join(secoes)


def carregar_dados() -> list[tuple[str, pd.DataFrame]]:
    arquivos = sorted(INPUT_DIR.glob("metricas_por_threshold*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {INPUT_DIR}.")

    dados = []
    for caminho in arquivos:
        label = source_label(caminho)
        df = carregar_csv(caminho)
        dados.append((label, df))
    return dados


def construir_html(dados: list[tuple[str, pd.DataFrame]]) -> str:
    resumo_html = "".join(resumo_por_origem(label, df) for label, df in dados)
    detalhes_html = detalhes_por_origem(dados)

    tabelas_metricas = "".join(
        f"""
        <section class="section">
            <div class="section-title">
                <div>
                    <h2>{html.escape(metrica.replace("_", " "))}</h2>
                    <p>Valores por threshold para cada avaliador.</p>
                </div>
            </div>
            <div class="panel">
                <div class="table-wrap">
                    {tabela_metricas_por_origem(dados, metrica)}
                </div>
            </div>
        </section>
        """
        for metrica in METRICAS
    )

    return f"""<!doctype html>
<html lang="pt-BR">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Comparacao de metricas dos avaliadores</title>
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
                max-width: 12ch;
            }}

            .subtitle {{
                margin: 0;
                max-width: 70ch;
                color: var(--muted);
                font-size: 1.03rem;
                line-height: 1.6;
            }}

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

            .summary-note {{ margin: 0; color: var(--muted); }}

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
                max-height: 640px;
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
                <div class="eyebrow">Comparacao de metricas dos avaliadores</div>
                <h1>Comparacao consolidada de resultados</h1>
                <p class="subtitle">
                    Este relatorio le todos os CSVs de {html.escape(INPUT_DIR.name)} e
                    organiza as metricas por threshold para comparacao direta.
                </p>
            </section>

            <section class="section">
                <div class="section-title">
                    <div>
                        <h2>Resumo por avaliador</h2>
                        <p>Melhor threshold pelo f1_verdadeiro em cada arquivo.</p>
                    </div>
                </div>
                <div class="summary-grid">
                    {resumo_html}
                </div>
            </section>

            {tabelas_metricas}

            <section class="section">
                <div class="section-title">
                    <div>
                        <h2>Detalhes completos</h2>
                        <p>Todas as linhas de cada CSV original.</p>
                    </div>
                </div>
                {detalhes_html}
            </section>

            <div class="footer">Arquivo gerado automaticamente a partir da pasta metricas.</div>
        </main>
    </body>
</html>
"""


def main() -> None:
    dados = carregar_dados()
    html_resultado = construir_html(dados)
    OUTPUT_HTML.write_text(html_resultado, encoding="utf-8")
    print(f"HTML gerado com sucesso: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()