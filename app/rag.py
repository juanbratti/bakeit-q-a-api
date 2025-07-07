# DOCS ============================================================
"""
    This file contains the RAG module for a dessert recipes retrieval system.
    In general, ituses:
        - OpenAI API for embedding and LLM models.
        - ChromaDB for the vector store.
        - LlamaIndex for the overall implementation.
        - LangDetect for language detection.
        - Dotenv for environment variables.
        - Logging for debug/info output.

    I used the following docs to build this rag system: 
    https://docs.llamaindex.ai/en/stable/understanding/rag/

    I also used the reranking techniques to retrieve more relevant results and don't overload the 
    llm with useless docs. Documentation:
    https://docs.llamaindex.ai/en/stable/examples/workflow/rag/
"""

# IMPORTS =========================================================

import chromadb
import os
from llama_index.llms.openai import OpenAI
# from llama_index.llms.cohere import Cohere
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Document,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.base.response.schema import Response
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from langdetect import detect

import re
import logging
from typing import Dict, List

# LOGGING SETUP FOR DEBUGGING PURPOSES ============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ENVIRONMENT VARIABLES ===========================================

load_dotenv()
# cohere_api_key = os.getenv("COHERE_API_KEY")
# if not cohere_api_key:
#     raise ValueError("COHERE_API_KEY is not set in environment variables")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")


# RAG CLASS ========================================================

class RAG:
    """Contains the methods needed to create the RAG system + configuration"""
    
    def __init__(self,      # I'd rather have a config file for these values
                 chroma_path: str = "./chromadb",
                 data_path: str = "./data",
                 collection_name: str = "dessert_recipes",
                 llm_model: str = "gpt-3.5-turbo", # cohere: "command-r-plus-04-2024"
                 temperature: float = 0.5,
                 embedding_model: str = "text-embedding-3-small", # cohere: "embed-english-v3.0",
                 top_k: int = 20,
                 top_n: int = 5, # this is for reranking
                 index_path: str = "./index_storage",
                 similarity_threshold: float = 0.3,
                 ):         # I keep them this way to follow the instructions of the project

        try:
            logger.info("Init of the RAG config")
            
            # basic validations, just in case
            if not os.path.exists(data_path):
                raise ValueError(f"Data not found: {data_path}")
            if not collection_name.strip():
                raise ValueError("Not a valid collection name")
            if top_k <= 0 or not (0 <= temperature <= 1):
                raise ValueError("Invalid top_k or temperature")

            # this is for later use in other methods of the class
            self.data_path = data_path
            self.index_path = index_path
            self.top_k = top_k
            self.top_n = top_n
            self.similarity_threshold = similarity_threshold

            # makes sure the paths to the chroma db & index exist
            os.makedirs(chroma_path, exist_ok=True)
            os.makedirs(index_path, exist_ok=True)

            # obtains the chroma client
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)

            # gets the chroma collection or creates it if it doesn't exist
            self.collection = self.chroma_client.get_or_create_collection(collection_name)
            logger.info(f"Chroma collection '{collection_name}' retrieved")

            # I use the Settings class from LlamaIndex to set the embedding & llm models.
            # this is just one layer more of abstraction to make it easier to change the models if needed.
            Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
            Settings.llm = OpenAI(model=llm_model, temperature=temperature, api_key=openai_api_key)

            # The following setting is to not have my custom chunks splitted by default
            Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=0)
            logger.info(f"language + embedding models set: {llm_model}, {embedding_model}")

            # sets the vector store with the corresponding chroma collection
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            # gets or creates the index
            if self._index_exists():
                # if the index is persisted and the chroma collection is not empty, loads it
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    persist_dir=self.index_path,
                )

                # this loads the index from the persisted storage
                self.index = load_index_from_storage(self.storage_context)
                logger.info("Existing index loaded from storage")
            else:
                # otherwise -> creates new storage context and ingests documents
                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                self._index_ingest()
                logger.info("New index created and documents ingested")
            
            # builds the query engine once during initialization
            self.query_engine = self._build_query_engine()
            logger.info("Query engine built and cached")
            
            logger.info("RAG system ready")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise ValueError(f"Initialization failed: {e}")

    def _index_exists(self) -> bool:
        """Checks if the index is persisted and collection is not empty"""
        # if the index is empty -> listdir is an empty list -> bool([]) = False
        # I also use the collection.count() just in case.
        return bool(os.listdir(self.index_path)) and self.collection.count() > 0
    
    def _load_recipes_from_txt(self) -> List[Document]:
        """This is for the preprocess of the txt file. Loads recipes from a text file 
        where each recipe is separated by a line of ==='s.
        
        This custom loader is only used for the specific data I found on the internet.
        It won't work with arbitrary txt files.
        """

        docs = []
        for file in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            recipes = re.split(r"\n={5,}\n", text)
            recipes_list = [recipe.strip() for recipe in recipes if recipe.strip()]
            docs.extend([Document(text=recipe) for recipe in recipes_list]) 
            # I don't add metadata because it adds more complexity to the code.
        
        return docs

    def _index_ingest(self) -> None:
        """Ingest the documents into the index. Assumes the data path is valid."""
        # loads the documents from the data path
        recipes_docs = self._load_recipes_from_txt()

        # if the txt file is empty, raises an error
        if not recipes_docs: 
            raise RuntimeError("Please provide recipes :(")
        
        # otherwise -> ingests the documents into the vector store
        self.index = VectorStoreIndex.from_documents(
            recipes_docs,
            storage_context=self.storage_context,
        )

        # the index is persisted in the index_path for later re-use
        self.index.storage_context.persist(persist_dir=self.index_path)

    def _translate_query(self, query: str) -> str:
        """In case the given query is NOT in english, this method translates it.
        This maintains consistency and allows for better retrieval of the recipes."""
    
        try:
            translation_prompt = f"""
                Translate the following query into English as accurately as possible.
                Return ONLY the translated text, no explanations, no additional commentary.
                Preserve the original meaning exactly, without adding, removing, or altering 
                any information.

                Query: {query}

                English translation:
            """
            translation = Settings.llm.complete(translation_prompt)
            result = str(translation).strip()
            
            if not result:
                raise RuntimeError("Translation returned empty result")
                
            logger.info(f"Translation done: '{query}' -> '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise RuntimeError(f"Translation failed: {e}")
    
    def _is_likely_english(self, text: str) -> bool:
        """Simple heuristic to check if text is likely English. 
        This saves a call to the model's api."""
        return detect(text) == 'en'
    
    def _get_prompt_template(self) -> PromptTemplate:
        """Returns my custom prompt template for the query.
        This is the prompt that the query engine will use to generate the response."""

        # custom prompt template
        return PromptTemplate(
              """You are a culinary assistant specialized in providing recipes based on a given context.
                Your task is to respond to user queries using ONLY the recipes available in the provided context.
                Your answer must be friendly and include the following information:

                1. Context: You will be given a list of recipes, each with its name, ingredients, and detailed instructions, including exact quantities and temperatures.
                2. Query: You will receive a query that may include keywords related to recipes or ingredients.

                Instructions:
                - Upon receiving the user's query, first search the names of the recipes in the context for matches with the query's keywords.
                - If you find recipes with matching keywords in their names, select those recipes and provide:
                1. Recipe Name
                2. Ingredients (including exact quantities)
                3. Instructions (step-by-step, including specific temperatures)
                - If no recipes match the keywords in the name, search for recipes that contain ingredients mentioned in the query.
                For those recipes, also provide:
                1. Recipe Name
                2. Ingredients (including exact quantities)
                3. Instructions (step-by-step, including specific temperatures)

                - Ensure your answer is clear, concise, and friendly. Use a welcoming and encouraging tone to motivate the user to try the recipes.

                Context:
                {{context_str}}

                Query:
                {{query_str}}
                """
        ) # just in case: context_str and query_str are passed internally by the query engine
    
    def _build_query_engine(self) -> RetrieverQueryEngine:
        """Builds the query engine with the retriever and the custom prompt.
        It uses the VectorIndexRetriever because I want to toggle the similarity threshold."""
    
        # configuration of the retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

        # configuration of the response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="compact", # documentation says it does less api calls 
            text_qa_template=self._get_prompt_template(), # usage of my custom prompt
        )

        # I apply reranking to get the top_n most relevant results
        reranker = LLMRerank(
            top_n=self.top_n,
            llm=Settings.llm,
        )

        # this is the query engine that will be used to query the index
        return RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                # I use the similarity threshold to filter out extreme non-relevant results + reranker
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=self.similarity_threshold),
                    reranker,
                ],

                # I could've used the reranker only, but sometimes (in random queries) some docs were 
                # retrieved anyway and passed to the LLM, which was a waste of tokens.
            )
    
    def _query_engine(self, english_query: str) -> Response: 
        """Returns the results of the query engine.
        This doesn't return the documents on their own, but rather the source nodes."""
        
        results = self.query_engine.query(english_query)
        return results
    
    def _get_documents(self, results: Response) -> List[str]:
        """Gets the documents from the results.
        This is used to return the documents on their own."""
        
        source_nodes = results.source_nodes
        documents = [node.node.get_content() for node in source_nodes]
        logger.info(f"amount of docs: {len(documents)}")
        return documents
    
    
    def query(self, query: str) -> Dict: # dict is interpreted as a json by fastapi
        """Main method to query the RAG system.
        This is the method that will be used to query the RAG system."""
        
        # basic validations
        original_query = query.strip()

        if not original_query:
            return {
                "query": query,
                "documents": [],
                "response": "Please provide a valid query."
            }
        
        try:
            logger.info(f"Processing query: '{original_query}'")
            
            # translate the query to English for better retrieval
            if not self._is_likely_english(original_query):
                english_query = self._translate_query(original_query)
                logger.info(f"Translated query: '{english_query}'")
            else:
                # If query is very likely already in English, skip translation
                english_query = original_query
                logger.info("Query is in English, skipping translation")

            # gets documents
            results = self._query_engine(english_query)
            documents = self._get_documents(results)
            
            # CASE 1: no relevant docs found
            if not documents:
                logger.info("No relevant documents found")
                return {
                    "query": query,
                    "documents": [],
                    "response": "No recipes found for your query :(",
                }
    
            # CASE 2: relevant docs found            
            # the response synthesizer already handles the response generation by the llm
            response_text = str(results.response)
            
            logger.info("Query processed successfully")
            return {
                "query": query,
                "documents": documents,
                "response": response_text,
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in query processing: {e}")
            return {
                "query": query,
                "documents": [],
                "response": "Sorry, an unexpected error occurred. Please try again."
            }
