import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Φόρτωση προεκπαιδευμένου μοντέλου spaCy με vectors
nlp = spacy.load('en_core_web_md')  # python -m spacy download en_core_web_md

texts_original = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn."
]

texts_paraphrased = [
    "Today is our dragon boat festival in our Chinese culture to celebrate it with all safe and great in our lives.",
    "During our final discussion, I told him about the new submission — the one we had been waiting for since last autumn."
]

# Υπολογισμός cosine similarity
for i, (orig, para) in enumerate(zip(texts_original, texts_paraphrased), 1):
    doc_orig = nlp(orig)
    doc_para = nlp(para)
    sim = cosine_similarity([doc_orig.vector], [doc_para.vector])[0][0]
    print(f"Κείμενο {i} - Cosine similarity (spaCy vectors): {sim:.4f}")

# Οπτικοποίηση PCA και t-SNE για λέξεις αρχικού κειμένου 1
words = [token.text for token in nlp(texts_original[0]) if token.has_vector and not token.is_punct]
vectors = np.array([token.vector for token in nlp(texts_original[0]) if token.has_vector and not token.is_punct])

# PCA
pca = PCA(n_components=2)
vecs_pca = pca.fit_transform(vectors)

# t-SNE με μικρότερο perplexity
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
vecs_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(18, 8))

# Σχεδιάγραμμα PCA
plt.subplot(1, 2, 1)
plt.scatter(vecs_pca[:, 0], vecs_pca[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (vecs_pca[i, 0], vecs_pca[i, 1]))
plt.title("PCA Projection of Word Embeddings (Original Text 1)")

# Σχεδιάγραμμα t-SNE
plt.subplot(1, 2, 2)
plt.scatter(vecs_tsne[:, 0], vecs_tsne[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (vecs_tsne[i, 0], vecs_tsne[i, 1]))
plt.title("t-SNE Projection of Word Embeddings (Original Text 1)")

plt.show()
