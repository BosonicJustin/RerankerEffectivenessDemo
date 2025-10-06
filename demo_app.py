"""
Interactive Web Demo: Compare Low Encoder vs Low Encoder + Reranker

This app allows you to:
1. Select k (number of results to show)
2. Enter a search query
3. See side-by-side comparison:
   - Left: Low encoder retrieves top-k directly
   - Right: Low encoder retrieves top-100, reranker picks top-k

How to run:
1. Make sure you have trained the reranker (trained_reranker/ directory exists)
2. Install: pip install streamlit
3. Run: streamlit run demo_app.py

Note: Models are cached after first load for faster subsequent runs.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from datasets import load_from_disk
import time

# Page config
st.set_page_config(
    page_title="Reranker Effectiveness Demo",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Reranker Effectiveness Demo")
st.markdown("### Compare: Low Encoder vs Low Encoder + Reranker")

# Load models and data (cached)
@st.cache_resource
def load_models():
    """Load encoder, reranker, and dataset"""
    low_encoder = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    reranker = CrossEncoder("./trained_reranker")
    test_dataset = load_from_disk("./test_dataset")

    # Create corpus
    corpus = [example["answer"] for example in test_dataset]

    # Encode corpus
    st.info("Encoding corpus... (this happens once)")
    corpus_embeddings = low_encoder.encode(corpus, convert_to_tensor=False, show_progress_bar=False)

    return low_encoder, reranker, corpus, corpus_embeddings

# Load everything
with st.spinner("Loading models and data..."):
    low_encoder, reranker, corpus, corpus_embeddings = load_models()

st.success(f"✅ Models loaded! Corpus size: {len(corpus)} documents")

# Sidebar controls
st.sidebar.header("Controls")
k = st.sidebar.slider("Select k (number of results)", min_value=1, max_value=20, value=5)
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("- **Left (Encoder only)**: Retrieve top-k with encoder")
st.sidebar.markdown("- **Right (Encoder + Reranker)**: Retrieve top-100 with encoder, rerank to get top-k")

# Example queries section (before search bar)
with st.expander("💡 Click to see example queries from the dataset", expanded=False):
    st.markdown("**Try these example questions from the GooAQ dataset:**")

    # Load actual questions from test dataset and randomly select
    test_dataset_for_examples = load_from_disk("./test_dataset")

    # Generate random sample (different each time the app restarts)
    if "example_indices" not in st.session_state:
        import random
        # No seed = truly random selection each time the app loads
        st.session_state.example_indices = random.sample(range(len(test_dataset_for_examples)), min(20, len(test_dataset_for_examples)))

    example_questions = [test_dataset_for_examples[i]["question"] for i in st.session_state.example_indices]

    for i, example_q in enumerate(example_questions):
        if st.button(f"🔍 {example_q}", key=f"example_btn_{i}"):
            st.session_state.query = example_q

# Search bar - use session state
if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Enter your search query:", value=st.session_state.query, placeholder="e.g., What is the capital of France?")

if query:
    st.markdown("---")

    # Create two columns
    col1, col2 = st.columns(2)

    # Encode query
    query_embedding = low_encoder.encode(query, convert_to_tensor=False)

    # LEFT SIDE: Low encoder only
    with col1:
        st.markdown("### 🔵 Low Encoder Only")
        st.caption("Directly retrieve top-k results")

        start = time.time()
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)[0]
        encoder_time = (time.time() - start) * 1000

        st.metric("Search Time", f"{encoder_time:.2f} ms")

        # Display results
        for i, hit in enumerate(hits, 1):
            score = hit['score']
            doc = corpus[hit['corpus_id']]

            with st.expander(f"**#{i}** - Score: {score:.4f}", expanded=(i <= 3)):
                st.write(doc)

    # RIGHT SIDE: Low encoder + reranker
    with col2:
        st.markdown("### 🟢 Low Encoder + Reranker")
        st.caption("Retrieve top-100, rerank to top-k")

        # Step 1: Retrieve top-100
        start = time.time()
        hits_100 = util.semantic_search(query_embedding, corpus_embeddings, top_k=100)[0]
        retrieval_time = (time.time() - start) * 1000

        # Step 2: Rerank top-100
        start = time.time()
        rerank_pairs = [[query, corpus[hit['corpus_id']]] for hit in hits_100]
        rerank_scores = reranker.predict(rerank_pairs)

        # Sort by reranker scores
        reranked = sorted(zip(hits_100, rerank_scores), key=lambda x: x[1], reverse=True)
        top_k_reranked = reranked[:k]
        rerank_time = (time.time() - start) * 1000

        total_time = retrieval_time + rerank_time

        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Retrieval", f"{retrieval_time:.2f} ms")
        with col2_2:
            st.metric("Reranking", f"{rerank_time:.2f} ms")
        with col2_3:
            st.metric("Total", f"{total_time:.2f} ms")

        # Display results
        for i, (hit, score) in enumerate(top_k_reranked, 1):
            doc = corpus[hit['corpus_id']]

            # Check if this result is in the left side top-k
            in_encoder_top_k = hit['corpus_id'] in [h['corpus_id'] for h in hits[:k]]
            emoji = "✨" if not in_encoder_top_k else "✅"

            with st.expander(f"{emoji} **#{i}** - Score: {score:.4f}", expanded=(i <= 3)):
                st.write(doc)
                if not in_encoder_top_k:
                    st.info("🔄 This result was NOT in encoder's top-k but was promoted by the reranker!")

    # Comparison metrics
    st.markdown("---")
    st.markdown("### 📊 Comparison")

    # Calculate overlap
    encoder_ids = set([hit['corpus_id'] for hit in hits[:k]])
    reranker_ids = set([hit['corpus_id'] for hit, _ in top_k_reranked])
    overlap = len(encoder_ids.intersection(reranker_ids))

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Overlap", f"{overlap}/{k}", f"{(overlap/k*100):.0f}%")
    with col_m2:
        st.metric("New results from reranker", f"{k - overlap}")
    with col_m3:
        speedup = encoder_time / total_time if total_time > 0 else 0
        st.metric("Speed comparison", f"{speedup:.2f}x" if speedup < 1 else f"{1/speedup:.2f}x slower")

else:
    st.info("👆 Enter a search query above or click on an example question to see the comparison!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>
Built with Streamlit | Low Encoder: static-retrieval-mrl-en-v1 | Reranker: Trained on GooAQ
</small>
</div>
""", unsafe_allow_html=True)
