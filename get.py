from haystack.nodes import QuestionGenerator, BM25Retriever, FARMReader
from haystack.nodes import EmbeddingRetriever

from haystack.pipelines import (
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,
    ExtractiveQAPipeline,
)
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import GenerativeQAPipeline
import openai
from haystack.utils import launch_es, print_questions
import json

document_store = InMemoryDocumentStore(use_bm25=True)
with open('points.json','r') as f:
    docs = json.loads(f.read())
contents = [i['content'] for i in docs]

# document_store.write_documents(docs)
import pandas as pd

from haystack.utils import fetch_archive_from_http


# Download
# Create embeddings for our questions from the FAQs
# In contrast to most other search use cases, we don't create the embeddings here from the content of our documents,
# but rather from the additional text field "question" as we want to match "incoming question" <-> "stored question".
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=False,
    scale_score=False,
)

embs = retriever.embed_queries(queries=contents).tolist()
for i in range(len(docs)):
    docs[i]['embedding'] = embs[i]
# Convert Dataframe to list of dicts and index them in our DocumentStore
document_store.write_documents(docs)

from haystack.pipelines import FAQPipeline

pipe = FAQPipeline(retriever=retriever)


def retrieve_answers(query):
    ans = pipe.run(query)['answers']
    print(ans)
    ans = [i.context for i in ans if i.score >= 0.30]
    return ans


def ask_gpt(query):
    openai.api_key = ""
    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",

            messages=[{"role": "user", "content": f"You are a city guide, give a short answer (less than 10 words) on {query}."}]
        )
    txt = completion['choices'][0]['message']['content']
    return txt

if __name__ == "__main__":
    print(retrieve_answers('Where is the shopping mall?'))
    print(ask_gpt("What are the main sights of Astana?"))
