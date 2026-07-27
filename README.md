# Movie Retrieval System

A movie information retrieval system that combines keyword and semantic search
to return relevant movie recommendations from a large database of film descriptions.

The system implements multiple search strategies — BM25 keyword search, dense
semantic search using sentence embeddings, and a hybrid Reciprocal Rank Fusion
(RRF) approach that combines both. It also supports LLM-powered query enhancement
and result reranking, along with multimodal search from images.

Evaluation is performed using standard IR metrics (Precision, Recall, F1, NDCG,
MRR, MAP) measured against a curated golden dataset.

## Keyword Search

- Good when user can write a very specific query,
eg. `Paddington` will return Paddington movies
- Bad when user cannot, eg. `funny movie` will return only results where
the words `funny` and `movie` appear in the title or description

- Start by building an inverted index of the database.
- Tokenize query and docs to compare tokens and return results
with higher token matches.
  - Remove stopwords, and 'stem' remaining words to increase relevant matches.
- bm25search command, which compares the
Term Frequency - Inverse Document Frequency (TF-IDF) of the query for each
doc in the database.
  - TF measures how many times a term appears in a doc.
  - IDF measures when a word is very rare in the dataset.
  - A high TF-IDF score represents a term being very common in a doc,
  but rare in the entire dataset.
  - Okapi BM25 smartly combines TF-IDF scores to return the best matches.

### Usage

- Run `uv run cli/keyword_search_cli.py search "<query>"` for standard keyword match search

  - eg. `uv run cli/keyword_search_cli.py search dinosaur`

    ```
    Searching for: dinosaur
    1. Jurassic Park
    2. Scooby-Doo and the Cyber Chase
    3. The Flintstones in Viva Rock Vegas
    4. The Good Dinosaur
    5. The Wizard
    ```

- Run `uv run cli/keyword_search_cli.py bm25search "<query>"`
  - Optional `--limit` flag returns custom number of search results.
  - eg. `uv run cli/keyword_search_cli.py search dinosaur`

    ```
    1. (3703) Anonymous Rex - Score: 11.56
    2. (54) Jurassic Park - Score: 11.32
    3. (1550) Carnosaur - Score: 10.84
    4. (977) The Wizard - Score: 9.96
    5. (1195) A Sound of Thunder - Score: 9.86
    ```

## Semantic Search

- Good when user is searching for a concept,
eg. `funny movie` will reliably return comedy movies.
- Bad when user is searching for something specific, eg.
`Monsters vs. Aliens` might return movies about aliens and monsters
rather than this specific movie the user is likely searching for.

- Build semantic embeddings database.
  - Pass doc database to a text embedder to encode semantic meanings.
  - This is a computationally heavy process, so we perform this once,
  and store the results.
- Semantically embed user query, and compare cosine similarity between
query embedding and each doc's embedding.
- Splitting our docs into chunks can prevent semantic dilution.
  - Semantic dilution occurs when a doc covers too many concepts to
  be carry all those concepts into just one embedding.
  - By including some overlap in our chunks, we can retain some of the
  context from the original doc.

### How to go further

Currently I store embeddings as a numpy file in a cache dir, and load these
into memory when performing search. This was an easy short-cut in implementation.
However, to make this production ready, this needs to be migrated to a
vector database.

Concepts studied:

- Locality Sensitive Hashing (LSH) - splitting vectors into 'buckets' to increase
search speed, at the potential cost of missing matches
- Hierarchical Navigable Small World (HNSW) - database consists of layers of
increasingly dense graphs. We find the local minimum at each layer to greatly
speed up search.
- Inverted File with Flat Vector (IVF) - vector space is divided in clusters
(similar to genres) and each cluster has a representative node. Comparison to
representative node can greatly reduce search space.

I would love to pursue these concepts further.

### Usage

- Run `uv run cli/semantic_search_cli.py search_chunked "<query>"`.
  - Optional `--limit` flag returns custom number of search results.
  - eg.`uv run cli/semantic_search_cli.py search_chunked "comedy dinosaur family"`

    ```
    1. The Good Dinosaur (score: 0.5372)
      65 million years ago, an asteroid made its way out of the belt and sped toward Earth. It turned red ...

    2. A Claymation Christmas Celebration (score: 0.5336)
      Situated in a facsimile of London's Christmas Square, the special is co-hosted by Rex (Johnny Counte...

    3. The Flintstones in Viva Rock Vegas (score: 0.5229)
      Young bachelors and best friends Fred Flintstone and Barney Rubble have recently qualified as crane ...

    4. Jeom-bak-i: Han-ban-do-eui Gong-ryong 3D (score: 0.4727)
      80 million years ago, during the Cretaceous period, a young Tarbosaurus named Speckles, for his uniq...

    5. The Wizard (score: 0.4711)
      Jimmy Woods is a young boy who has suffered from an unnamed, but serious mental disorder ever since ...
    ```

## Hybrid Search

- Keyword search and semantic search both have their pros and cons...
- We can combine their results by normalizing their scores.
- Min-max normalization:
  - Order the results from each search method and normalizing the scores.
  - Combine normalized score from both methods into hybrid score for each doc.
- Reciprocal rank fusion:
  - Better handles major outliers than standard min-max.

### Usage

- Run `uv run cli/hybrid_search_cli.py weighted-search "<query>" --alpha`.
  - Optional `--limit` flag returns custom number of search results.
  - `--alpha` flag controls weight given to keyword search scores.
  - eg. `uv run cli/hybrid_search_cli.py weighted-search "funny dinosaur movie Jurassic Park" --alpha 0.6`

    ```
    SemanticResult]:
        Hybrid Score: 0.9674
        BM25: 1.0000, Semantic: 0.9184
        Description: Industrialist John Hammond and his bioengineering company, InGen, have created a theme park called J...
    2. Lost River
        Hybrid Score: 0.6010
        BM25: 0.4398, Semantic: 0.8427
        Description: A young boy runs out of a house into the tall grass outside. We see a slow montage of a crumbling ci...
    3. The Good Dinosaur
        Hybrid Score: 0.5328
        BM25: 0.2213, Semantic: 1.0000
        Description: 65 million years ago, an asteroid made its way out of the belt and sped toward Earth. It turned red ...
    4. Carnosaur
        Hybrid Score: 0.5302
        BM25: 0.3791, Semantic: 0.7568
        Description: In a small town in the American Southwest, a mysterious illness befalls its citizens. Dr. Jane Tiptr...
    5. Ice Age: Dawn of the Dinosaurs
        Hybrid Score: 0.4909
        BM25: 0.2944, Semantic: 0.7857
        Description: Ellie (Queen Latifah) and Manny (Ray Romano) are expecting their first child, and Manny is obsessed ...
    ```

- Run `uv run cli/hybrid_search_cli.py rrf-search "<query>"`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight)
  - eg. `uv run cli/hybrid_search_cli.py rrf-search "funny dinosaur movie Jurassic Park"`

    ```
    1. Jurassic Park
        RRF Score: 0.0325
        BM25 Rank: 1, Semantic Rank: 2
        Description: Industrialist John Hammond and his bioengineering company, InGen, have created a theme park called J...
    2. Lost River
        RRF Score: 0.0318
        BM25 Rank: 2, Semantic Rank: 4
        Description: A young boy runs out of a house into the tall grass outside. We see a slow montage of a crumbling ci...
    3. Carnosaur
        RRF Score: 0.0294
        BM25 Rank: 7, Semantic Rank: 9
        Description: In a small town in the American Southwest, a mysterious illness befalls its citizens. Dr. Jane Tiptr...
    4. The Wizard
        RRF Score: 0.0287
        BM25 Rank: 3, Semantic Rank: 18
        Description: Jimmy Woods is a young boy who has suffered from an unnamed, but serious mental disorder ever since ...
    5. A Sound of Thunder
        RRF Score: 0.0284
        BM25 Rank: 10, Semantic Rank: 11
        Description: In Chicago, 2055, the Time Safari company offers the ability for people to hunt dinosaurs in the pas...
    ```

## LLM

We can also introduce LLMs to improve our search:

- Enhancing queries by fixing spelling mistakes or improving specificity.
- Reranking results.

### Usage

For all LLM commands, user must store `GEMINI_API_KEY=` in a `.env` file
in the project root.

- Run `uv run cli/hybrid_search_cli.py rrf-search "<query>"`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight)
  - Optional `--enhance` flag can either spell, rewrite, or expand a user query
  - Optional `--rerank-method` flag can ask an LLM to
  either individually or batch re-rank results
  - eg. `uv run cli/hybrid_search_cli.py rrf-search "funny dinosaur movie Jurassic Park" --rerank-method batch`

  ```
    1. Jurassic Park
        Rerank Score: 1
        RRF Score: 0.0325
        BM25 Rank: 1, Semantic Rank: 2
        Description: Industrialist John Hammond and his bioengineering company, InGen, have created a theme park called J...
    2. Amusement
        Rerank Score: 2
        RRF Score: 0.0241
        BM25 Rank: 49, Semantic Rank: 7
        Description: During the opening credits, pictures of three girls when they were children, as adolescent, and as y...
    3. Pee-wee's Big Adventure
        Rerank Score: 3
        RRF Score: 0.0279
        BM25 Rank: 8, Semantic Rank: 16
        Description: Pee-wee Herman has a heavily accessorized bicycle that he treasures and that his neighbor, Francis B...
    4. Bringing Up Baby
        Rerank Score: 4
        RRF Score: 0.0247
        BM25 Rank: 18, Semantic Rank: 24
        Description: David Huxley (Cary Grant) is a mild-mannered paleontologist. For the past four years, he has been tr...
    5. Ice Age: Dawn of the Dinosaurs
        Rerank Score: 5
        RRF Score: 0.0281
        BM25 Rank: 17, Semantic Rank: 6
        Description: Ellie (Queen Latifah) and Manny (Ray Romano) are expecting their first child, and Manny is obsessed ...
  ```

## Cross-Encoder

LLM API calls can be very slow and expensive. A cross-encoder is much cheaper,
faster, and more specialised than a general purpose LLM. It semantically embeds
queries and docs and outputs a single similarity score.

### Usage

- Run `uv run cli/hybrid_search_cli.py rrf-search "<query>" --rerank-method cross_encoder`.
  - Optional `--limit` flag returns custom number of search results.
  - Optional `--k` flag controls weight given to lower/higher scores.
  (low k means low scores given less weight).
  - eg. `uv run cli/hybrid_search_cli.py rrf-search "funny dinosaur movie Jurassic Park" --rerank-method cross_encoder`

  ```
    1. Jurassic Park
        Rerank Score: 0.8303611278533936
        RRF Score: 0.0325
        BM25 Rank: 1, Semantic Rank: 2
        Description: Industrialist John Hammond and his bioengineering company, InGen, have created a theme park called J...
    2. Anonymous Rex
        Rerank Score: -8.348739624023438
        RRF Score: 0.0270
        BM25 Rank: 4, Semantic Rank: 28
        Description: In an alternate timeline, dinosaurs have managed to survive the KT Extinction Event and now live amo...
    3. Ice Age: Dawn of the Dinosaurs
        Rerank Score: -8.757635116577148
        RRF Score: 0.0281
        BM25 Rank: 17, Semantic Rank: 6
        Description: Ellie (Queen Latifah) and Manny (Ray Romano) are expecting their first child, and Manny is obsessed ...
    4. When Dinosaurs Ruled the Earth
        Rerank Score: -9.080493927001953
        RRF Score: 0.0261
        BM25 Rank: 11, Semantic Rank: 23
        Description: A tribe on a cliff are about to sacrifice three blonde women. Three priests, wearing dinosaur hides,...
    5. Pee-wee's Big Adventure
        Rerank Score: -9.413055419921875
        RRF Score: 0.0279
        BM25 Rank: 8, Semantic Rank: 16
        Description: Pee-wee Herman has a heavily accessorized bicycle that he treasures and that his neighbor, Francis B...
  ```

## Evaluation Metrics

The system is evaluated against a curated golden dataset of 10 queries, each
with a set of ground-truth relevant movies. The following standard IR metrics
are computed at cutoff k=5:

| Metric | Description |
|---|---|
| **Precision@k** | Fraction of retrieved results that are relevant |
| **Recall@k** | Fraction of all relevant docs that were retrieved |
| **F1@k** | Harmonic mean of Precision and Recall |
| **NDCG@k** | Normalised Discounted Cumulative Gain — rewards relevant docs ranked higher |
| **MRR** | Mean Reciprocal Rank — how high the first relevant result appears |
| **MAP@k** | Mean Average Precision — area under the precision-recall curve |

### Benchmark Results (RRF hybrid search, k=5)

```
----------------------------------------------------------------------------------------------------
  AGGREGATE SUMMARY  (macro-averaged over 10 queries, k=5)
----------------------------------------------------------------------------------------------------
Query                                   Precision@5  Recall@5      F1@5    NDCG@5       MRR     MAP@5
-----------------------------------------------------------------------------------------------------
cute british bear marmalade                 0.8000    0.8000    0.8000    0.7860    1.0000    0.6433
talking teddy bear comedy                   0.8000    0.8000    0.8000    0.6608    0.5000    0.5433
children's animated bear adventure          0.8000    0.8000    0.8000    0.8688    1.0000    0.8000
friendship transformation magic with ...    0.8000    0.8000    0.8000    0.8539    1.0000    0.7600
dinosaur park                               1.0000    1.0000    1.0000    1.0000    1.0000    1.0000
wizards and magic                           0.8000    0.8000    0.8000    0.8688    1.0000    0.8000
superhero saves the world                   0.8000    0.8000    0.8000    0.8688    1.0000    0.8000
zombie apocalypse                           0.8000    0.8000    0.8000    0.8688    1.0000    0.8000
car racing                                  0.8000    0.8000    0.8000    0.8688    1.0000    0.8000
romantic comedy wedding                     0.8000    0.8000    0.8000    0.8688    1.0000    0.8000
-----------------------------------------------------------------------------------------------------
MACRO AVERAGE                               0.8200    0.8200    0.8200    0.8514    0.9500    0.7747
----------------------------------------------------------------------------------------------------
```

### Usage

- Run `uv run cli/evaluation_cli.py`
  - Optional `--limit` flag sets the number of results to retrieve per query (default: 5)
  - Optional `--search-type` flag selects `rrf` (default), `bm25`, or `semantic`
  - Optional `--enhance` and `--rerank-method` flags enable LLM query enhancement / reranking

  eg. `uv run cli/evaluation_cli.py --limit 10 --search-type rrf`

  ```
  ============================================================
  Movie Retrieval System — IR Evaluation
  ============================================================
    Search type  : rrf
    k (cut-off)  : 10
    Enhancement  : none
    Reranking    : none
  ============================================================

  Query: "dinosaur park"
    Retrieved  : ['Jurassic Park', 'Lost River', 'Carnosaur', ...]
    Hits       : ['Jurassic Park', 'Lost River', 'Carnosaur', ...]
    Precision@10: 1.0000
    Recall@10   : 1.0000
    NDCG@10     : 1.0000
    RR          : 1.0000
  ```

## Retrieval Augmented Generation (RAG)

1. **Retrieve** relevant documents using standard search algorithms.
2. **Augment** the LLMs context with the most relevant documents.
3. **Generate** a natural language response to the user query.

Implemented here are the options for standard RAG, summarizing (with citations),
or answering a question.

### Usage

- Run `uv run cli/augmented_generation_cli.py <cmd> "<query>"` where cmd is one of:
  - `rag`
  - `summarize`
  - `citations`
  - `question`

  - eg. `uv run cli/augmented_generation_cli.py question 'Who are the main characters in Jurassic Park?'`

  ```
    Search Results:
    -  Jurassic Park
    -  Mike and Dave Need Wedding Dates
    -  House II: The Second Story
    -  The Last of the Finest
    -  Tokyo Babiron

    Answer:
    The main characters in Jurassic Park are industrialist John Hammond, lawyer Donald Gennaro, mathematician Ian Malcolm, paleontologist Dr. Alan Grant, paleobotanist Dr. Ellie Sattler, and Hammond's grandchildren, Lex and Tim Murphy. Other significant characters include computer programmer Dennis Nedry, park game warden Robert Muldoon, and chief engineer Ray Arnold.
  ```

## Multimodal Search

Also included is the ability to provide an image path and perform a search,
either purely from that image, or with an attached text query. The image with the
attached query is passed to an LLM, which rewrites the query based on the image.
A pure image query, on the other hand, is semantically embedded by a multimodal
model. This model is capable of semantically embedding both images and text.
This is made possible by a contrastive learning approach to its training.


### Usage

- Run `uv run cli/describe_image_cli.py --image "<path_to_image>" --query "<query>"`
  - eg. `uv run cli/describe_image_cli.py image_search --image "data/paddington.jpeg" --query "what movie is this charatcer from?"`

    ```
    Rewritten query: Paddington Bear in blue coat and red hat movie
    ```

- Run `uv run cli/multimodal_search_cli.py image_search "<image_path>"`
  - eg. `uv run cli/multimodal_search_cli.py image_search "data/paddington.jpeg"`

    ```
    1. Paddington (similarity: 0.309)
        Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,...

    2. Ted (similarity: 0.294)
        In 1985, eight-year-old John Bennett makes a Christmas wish that his teddy bear, Ted, would come to ...

    3. Sing (similarity: 0.275)
        Set in an animated world controlled by animals, the film starts with a young koala bear named Buster...

    4. Bastille Day (similarity: 0.269)
        On the eve of Bastille Day in Paris, an American conman Michael Mason, steals a woman's handbag with...

    5. Guardians of the Galaxy (similarity: 0.269)
        On planet Earth in 1988, young Peter Quill (Wyatt Oleff) sits in the waiting room of a hospital, lis...
    ```
