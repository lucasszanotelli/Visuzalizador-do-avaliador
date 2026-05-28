import logging

from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

_MODEL = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
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

def chatbot_faq_score(faq_answer: str, chatbot_answer: str) -> dict:
    semantic_score = semantic_similarity(faq_answer, chatbot_answer)
    lexical_score = lexical_similarity(faq_answer, chatbot_answer)

    final_score = 0.8 * semantic_score + 0.2 * lexical_score

    return {
        # "faq_answer": faq_answer,
        # "chatbot_answer": chatbot_answer,
        "semantic_score": round(semantic_score, 4),
        "lexical_score": round(lexical_score, 4),
        "final_score": round(final_score, 4)
    }   
DEFAULT_FAQ_TEXT = (
    "O Cadastro Ambiental Rural – CAR é um registro público eletrônico nacional, "
    "obrigatório para todos os imóveis rurais, com a finalidade de integrar as "
    "informações ambientais das propriedades e posses rurais, compondo base de "
    "dados para controle, monitoramento, planejamento ambiental e econômico e "
    "combate ao desmatamento. Foi criado pela Lei 12.651/2012, art. 29."
)

DEFAULT_CHATBOT_TEXT = (
    "Cadastro Ambiental Rural – CAR é um registro eletrônico de abrangência "
    "nacional, obrigatório para todos os imóveis rurais. Finalidade do CAR: "
    "Integrar as informações ambientais das propriedades e posses rurais."
    "Compor uma base de dados para:  Controle ambiental.Monitoramento."
    "Planejamento ambiental e econômico.Combate ao desmatamento.Definição "
    "detalhada: O CAR contém dados do proprietário ou possuidor rural, planta "
    "georreferenciada do perímetro do imóvel, áreas de interesse social e de "
    "utilidade pública.Inclui a localização dos remanescentes de vegetação "
    "nativa, Áreas de Preservação Permanente, Áreas de Uso Restrito, áreas "
    "consolidadas e Reservas Legais. Em resumo, o CAR é um instrumento essencial "
    "para a gestão e regularização ambiental dos imóveis rurais no Brasil. Se "
    "precisar de mais detalhes, estou à disposição!"
)


if __name__ == "__main__":
    resultado = chatbot_faq_score(DEFAULT_FAQ_TEXT, DEFAULT_CHATBOT_TEXT)

    semantic_score = resultado["semantic_score"]
    lexical_score = resultado["lexical_score"]
    final_score = resultado["final_score"]

    # Imprimindo os valores
    print(f"\nSemelhança Semântica: {semantic_score}")
    print(f"\nSemelhança Lexical: {lexical_score}")
    print(f"\nPontuação Final: {final_score}")

    # Como interpretar
    # 0.85 a 1.00 -> excelente
    # 0.70 a 0.84 -> aceitavel
    # 0.50 a 0.69 -> fraco
    # abaixo de 0.50 -> ruim