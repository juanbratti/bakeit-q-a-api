# DOCS ============================================================

"""
This file contains the orchestrator agent for the dessert recipe assistant.
I used the LlamaIndex Pattern 2 of Multi-agent workflows.
Docs: https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#pattern-2--orchestrator-agent-sub-agents-as-tools

The agent has the following tools:
- is_english: checks if the query is in English
- translate_query: translates the query to English
- get_documents: retrieves the documents from the RAG index
- generate_response: generates the response to the query

For this implementation, I'm assuming the index and chroma db already exist.
"""

# IMPORTS =========================================================

import os
import chromadb
from typing import List, Dict
from dotenv import load_dotenv
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from langdetect import detect

# ENVIRONMENT VARIABLES =============================================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# LOGGING =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ORCHESTRATOR CLASS ===============================================

class AgentOrchestrator:
    def __init__(self, # hyperparameters, to make it easier to change if needed
                chroma_path: str = "./chromadb",
                index_path: str = "./index_storage",
                collection_name: str = "dessert_recipes",
                llm_model: str = "gpt-3.5-turbo",
                embedding_model: str = "text-embedding-3-small",
                top_k: int = 5,
            ):
        
        Settings.llm = OpenAI(model=llm_model, api_key=openai_api_key)
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
        self.top_k = top_k

        # --- the following assumes the index and chroma db already exist! --- #
        # initializes ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(collection_name)
        
        # loads the vector store with the ChromaDB collection
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        
        # loads the index from the persisted storage
        self.storage_context = StorageContext.from_defaults(
            persist_dir=index_path,
            vector_store=self.vector_store,
        )
        self.index = load_index_from_storage(self.storage_context)

        # builds orchestrator agent
        self.agent = self._build_orchestrator()
        logger.info("AgentOrchestrator initialized successfully")

    def _build_orchestrator(self) -> FunctionAgent:
        """Builds the orchestrator agent.
            I used the FunctionAgent class from LlamaIndex.
            It has the following tools:
            - is_english: checks if the query is in English
            - translate_query: translates the query to English
            - get_documents: queries the index and retrieves the documents
            - generate_response: generates the response for the query
            I used the Context class from LlamaIndex to pass the state to the tools.
        """
        
        # TOOLS =====================================================
        
        async def is_english(ctx: Context, query: str) -> bool:
            """Simple heuristic to check if text is likely English. This saves a call to the OpenAI api"""

            try:
                is_english = detect(query) == 'en'
                logger.info(f"Is English: {is_english}")

                # save information in the state
                state = await ctx.store.get("state")
                state["is_english"] = is_english
                state["original_query"] = query
                await ctx.store.set("state", state)

                return is_english
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                return None

        async def translate_query(ctx: Context) -> str:
            """Translates the query given by the user to English because that's the language
            the recipes were embedded in. This maintains consistency and allows for better
            retrieval of the recipes.
            
            translated_query has the original query if the query is already in English,
            otherwise it has the translated query."""

            state = await ctx.store.get("state")
            original_query = state.get("original_query")

            # if the query is already in English, returns it
            if state.get("is_english"):
                english_query = original_query
                logger.info("Query is already in English")
            else:
                # if the query is not in English, translates it
                translation_prompt = f"""
                    Translate the following query into English as accurately as possible.
                    Return ONLY the translated text, no explanations, no additional commentary.
                    Preserve the original meaning exactly, without adding, removing, or altering 
                    any information.

                    Query: {original_query}

                    English translation:
                """ # same prompt as the one used in the RAG system
                
                translation = Settings.llm.complete(translation_prompt)
                english_query = str(translation).strip()
                
                logger.info(f"Translation done: '{original_query}' -> '{english_query}'")

            # save information in the state
            state["translated_query"] = english_query
            await ctx.store.set("state", state)

            return english_query
        
        async def get_documents(ctx: Context) -> List[str]:
            """Retrieves documents from the vector store based on the query"""

            state = await ctx.store.get("state")

            english_query = state.get("translated_query")

            logger.info(f"Retrieving documents for query: '{english_query}'")

            # query the index and retrieve the documents
            results = self.index.as_query_engine(similarity_top_k=self.top_k).query(english_query)
            source_nodes = results.source_nodes
            documents = [node.node.get_content() for node in source_nodes]

            logger.info(f"Retrieved {len(documents)} documents")

            # save information in the state
            state["documents"] = documents
            await ctx.store.set("state", state)

            return documents

        async def generate_response(ctx: Context) -> str:
            """Generates the final response using the retrieved documents"""

            state = await ctx.store.get("state")
            documents = state.get("documents")

            if not documents:
                response_text = "I couldn't find any relevant recipes for your query."
                state["answer"] = response_text
                await ctx.store.set("state", state)
                return response_text
            
            english_query = state.get("translated_query")

            prompt = f"""
                Answer the following question: {english_query} using the documents below.

                Here are the documents that may contain the answer:
                {documents}

                Please provide a helpful response based on the documents above. 
                If the documents don't contain relevant information, say so.
            """

            # generate the response
            response = Settings.llm.complete(prompt=prompt)
            response_text = str(response).strip()

            # save information in the state
            state["answer"] = response_text
            await ctx.store.set("state", state)

            logger.info("Response generated successfully")
            return response_text

        tools = [
            is_english,
            translate_query,
            get_documents,
            generate_response,
        ]

        system_prompt = (
            "You are an orchestrator agent for a dessert recipe assistant. "
            "You must coordinate tools to detect language, translate queries if needed, "
            "retrieve recipes from a vector store, and generate a final response. "
            "Follow this workflow: 1) Check if query is English, 2) Translate if needed, "
            "3) Get relevant documents, 4) Generate response based on documents. "
            "Only use available tools. Once you've generated the response, return it."
        )

        return FunctionAgent(
            tools=tools,
            llm=Settings.llm,
            system_prompt=system_prompt,
            initial_state={
                "is_english": None,
                "original_query": None,
                "translated_query": None,
                "query_results": None,
                "documents": [],
                "answer": None
            }
        )

    async def run(self, query: str) -> str:
        """Run the orchestrator agent asynchronously."""

        # basic validation
        query_strip = query.strip()
        if not query_strip:
            return "Please provide a valid query."

        try:
            logger.info(f"Running orchestrator for query: '{query_strip}'")
            response = await self.agent.run(query_strip)
            logger.info(f"Orchestrator response generated successfully")
            
            # returns the agent's response directly
            return str(response)
        except Exception as e:
            logger.error(f"Orchestrator run failed: {e}")
            return f"Sorry, an error occurred: {str(e)}"
