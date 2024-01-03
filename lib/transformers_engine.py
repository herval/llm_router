from lib import Engine

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError(
        'Please install sentence-transformers with pip install sentence-transformers to use the Sentence Transformers Engine')


class SentenceTransformersEngine(Engine):
    def __init__(self, model_name='all-distilroberta-v1', threshold=0.0):
        """
        :param model_name: sentence_transformer model to use
        :param threshold: minimum similarity score to match a route
        """
        self.sentences = []
        self.routes_by_sentence = {}
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def encode(self, routes):
        for r in routes:
            for s in r.sentences:
                self.routes_by_sentence[s] = r.name
                self.sentences.append(s)
        self.embeddings = self.model.encode(self.sentences, convert_to_tensor=True)

    def match(self, query):
        embeddings2 = self.model.encode([query], convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = util.cos_sim(self.embeddings, embeddings2)

        # Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(len(cosine_scores) - 1):
            pairs.append({'index': i, 'score': cosine_scores[i][0]})

        # Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        # print(pairs[0:3])
        # print(COMMAND_EMBEDDINGS[self.sentences[pairs[0]['index']]])

        if pairs[0]['score'] > self.threshold:  # first one likely
            idx = pairs[0]['index']
            return self.routes_by_sentence[self.sentences[idx]]
