from flashrank import Ranker, RerankRequest

ranker = Ranker()
query = "What is the capital of France?"
passages = [
    {"id": 1, "text": "Paris is the capital of France.", "meta": {"source": "doc1"}},
    {"id": 2, "text": "London is the capital of the UK.", "meta": {"source": "doc2"}},
    {"id": 3, "text": "France is a country in Europe.", "meta": {"source": "doc3"}},
]

request = RerankRequest(query=query, passages=passages)
results = ranker.rerank(request)

print(results)
