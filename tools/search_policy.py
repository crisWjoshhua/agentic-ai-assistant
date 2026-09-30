from RAG.retriever import retrieve


def search_policy(query: str) -> str:
    """
    Searches the company policy documents using Chroma
    and returns relevant information with source information.
    """

    results = retrieve(
        query,
        k=3,
        threshold=1.0
    )

    if not results:
        return (
            "No sufficiently relevant information was found "
            "in the policy documents."
        )

    formatted_results = []

    for result in results:

        formatted_results.append(
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n\n"
            f"{result['text']}"
        )

    return "\n\n-------------------------\n\n".join(
        formatted_results
    )