import os
import asyncio
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph

# --- CRITICAL FIX: Correct Imports for GraphRetriever and Eager from the dedicated package ---
# from langchain_graph_retriever import GraphRetriever
# from langchain_graph_retriever.strategies import Eager # This is the correct path for Eager
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain.retrievers import MergerRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure VectorStore is imported for type hinting
from langchain_core.vectorstores import VectorStore

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field, PrivateAttr



class MergerRetriever(BaseRetriever):
    retrievers: List[BaseRetriever] = Field(...)

    def _fix_page_content(self, docs: List[Document]) -> List[Document]:
        fixed_docs = []
        for doc in docs:
            content = doc.page_content
            if isinstance(content, list):
                content = "\n".join(content)
            fixed_docs.append(Document(page_content=content, metadata=doc.metadata))
        return fixed_docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for r in self.retrievers:
            docs = r.get_relevant_documents(query)
            all_docs.extend(docs)
        return self._fix_page_content(all_docs)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for r in self.retrievers:
            docs = await r.ainvoke(query)
            all_docs.extend(docs)
        return self._fix_page_content(all_docs)


class MCP_ChatBot:
    def __init__(self, llm_model_name="gemma3:27b", embedding_model_name="NeuML/pubmedbert-base-embeddings"):
        print("MCP_ChatBot.py - Version 2025-06-27-EagerFix-AttemptFinal-Revised-v4-CorrectGraphRetriever") # Version Identifier

        # --- 1. Ollama LLM Configuration (for answering) ---
        print("Initializing LLM...")
        self.llm = ChatOllama(model=llm_model_name, base_url="http://localhost:11434", temperature=0.0)
        print("LLM Initialized.")

        # --- 2. HuggingFace Embedding Model Configuration (for Neo4jVector) ---
        print(f"Initializing HuggingFace Embeddings for Neo4jVector: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("HuggingFace Embeddings Initialized.")

        # --- 3. Neo4j Connection Details ---
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://digbio-xugpu-3.missouri.edu:7687")
        NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ZTjgBxsK3KczbEq6uyTa")
        print(f"Neo4j URI: {NEO4J_URI}, Username: {NEO4J_USERNAME}")

        # --- 4. Initialize Neo4jGraph for schema understanding and traversal ---
        self.neo4j_graph = None
        try:
            self.neo4j_graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )
            self.neo4j_graph.refresh_schema()
            print("Neo4jGraph connection established and schema refreshed.")
        except Exception as e:
            print(f"ERROR: Could not establish Neo4jGraph connection or refresh schema: {e}")
            print("Graph traversal will be limited or fail without a working Neo4jGraph instance. This is a critical error.")
            raise e

        print("Attempting to connect to Neo4jVector store(s)...")

        self.individual_retrievers = []

        try:
            # --- NODE VECTOR STORE CONFIGURATIONS ---

            # Assuming you have self.embeddings, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD defined

            # 1 Gene Nodes
            print("Configuring Gene Neo4jVector store...")
            gene_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="gene_embeddings",
                node_label="Gene",
                text_node_properties=["name", "symbol", "position", "link", "id"],
                embedding_node_property="embedding",
            )
            print("Initialized Gene Neo4jVector store.")

            # 2 Ortholog Nodes
            print("Configuring Ortholog Neo4jVector store...")
            ortholog_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="ortholog_embeddings",
                node_label="Ortholog",
                text_node_properties=["symbol", "link", "id", "name"],
                embedding_node_property="embedding",
            )
            print("Initialized Ortholog Neo4jVector store.")

            # 3 Reaction Nodes
            print("Configuring Reaction Neo4jVector store...")
            reaction_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="reaction_embeddings",
                node_label="Reaction",
                text_node_properties=["equation", "name", "link", "id", "definition"],
                embedding_node_property="embedding",
            )
            print("Initialized Reaction Neo4jVector store.")

            # 4 Compound Nodes
            print("Configuring Compound Neo4jVector store...")
            compound_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="compound_embeddings",
                node_label="Compound",
                text_node_properties=["names", "formula", "link", "id"],
                embedding_node_property="embedding",
            )
            print("Initialized Compound Neo4jVector store.")

            # 5 Pathway Nodes
            print("Configuring Pathway Neo4jVector store...")
            pathway_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="pathway_embeddings",
                node_label="Pathway",
                text_node_properties=["link", "id", "title", "image", "name"],
                embedding_node_property="embedding",
            )
            print("Initialized Pathway Neo4jVector store.")

            # 6 PathwayEntry Nodes
            print("Configuring PathwayEntry Neo4jVector store...")
            pathway_entry_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="pathwayentry_embeddings",
                node_label="PathwayEntry",
                text_node_properties=["id", "type", "name"],
                embedding_node_property="embedding",
            )
            print("Initialized PathwayEntry Neo4jVector store.")

            # 7 EC Nodes
            print("Configuring EC Neo4jVector store...")
            ec_vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="ec_embeddings",
                node_label="EC",
                text_node_properties=["sysname", "link", "id", "names"],
                embedding_node_property="embedding",
            )
            print("Initialized EC Neo4jVector store.")

            # --- RELATIONSHIP VECTOR STORE CONFIGURATIONS ---

            # Assuming you have self.embeddings, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD defined

                        # 1 CONTAINS Relationships
            print("Configuring CONTAINS Relationship Neo4jVector store...")
            contains_vector_store   = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="contains_embeddings",
                embedding_node_property="embedding",
                # retrieval_query="""
                #     CALL db.index.vector.queryRelationships($index, $k, $embedding)
                #     YIELD relationship, score
                #     MATCH (startNode)-[relationship]->(endNode)
                #     RETURN
                #         'Relationship ' + type(relationship) +
                #         ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                #         ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') +
                #         CASE
                #             WHEN relationship.type IS NOT NULL THEN ' (Type: ' + relationship.type + ')'
                #             ELSE ''
                #         END AS text,
                #         score,
                #         relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                # """
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' +
                        CASE WHEN apoc.meta.type(startNode.name) = 'LIST' THEN apoc.text.join(startNode.name, ', ') ELSE COALESCE(startNode.name, startNode.id, 'Unknown Node') END +
                        ' to ' +
                        CASE WHEN apoc.meta.type(endNode.name) = 'LIST' THEN apoc.text.join(endNode.name, ', ') ELSE COALESCE(endNode.name, endNode.id, 'Unknown Node') END +
                        CASE
                            WHEN apoc.meta.type(relationship.type) = 'LIST' THEN ' (Type: ' + apoc.text.join(relationship.type, ', ') + ')'
                            WHEN relationship.type IS NOT NULL THEN ' (Type: ' + relationship.type + ')'
                            ELSE ''
                        END AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized CONTAINS GraphRetriever.")

            # 2 BELONGS_TO Relationships
            print("Configuring BELONGS_TO Relationship Neo4jVector store...")
            belongs_to_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="belongs_to_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized BELONGS_TO GraphRetriever.")

            # 3 CATALYZES Relationships
            print("Configuring CATALYZES Relationship Neo4jVector store...")
            catalyzes_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="catalyzes_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized CATALYZES GraphRetriever.")

            # 4 PRODUCES Relationships
            print("Configuring PRODUCES Relationship Neo4jVector store...")
            produces_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="produces_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized PRODUCES GraphRetriever.")

            # 5 ENCODES Relationships
            print("Configuring ENCODES Relationship Neo4jVector store...")
            encodes_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="encodes_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized ENCODES GraphRetriever.")

            # 6 SUBSTRATE_OF Relationships
            print("Configuring SUBSTRATE_OF Relationship Neo4jVector store...")
            substrate_of_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="substrate_of_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized SUBSTRATE_OF GraphRetriever.")

            # 7 MEMBER_OF Relationships
            print("Configuring MEMBER_OF Relationship Neo4jVector store...")
            member_of_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="member_of_embeddings",
                embedding_node_property="embedding",
                # retrieval_query="""
                #     CALL db.index.vector.queryRelationships($index, $k, $embedding)
                #     YIELD relationship, score
                #     MATCH (startNode)-[relationship]->(endNode)
                #     RETURN
                #         'Relationship ' + type(relationship) +
                #         ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                #         ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') +
                #         CASE
                #             WHEN relationship.type IS NOT NULL THEN ' (Type: ' + relationship.type + ')'
                #             ELSE ''
                #         END AS text,
                #         score,
                #         relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                # """
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' +
                        CASE WHEN apoc.meta.type(startNode.name) = 'LIST' THEN apoc.text.join(startNode.name, ', ') ELSE COALESCE(startNode.name, startNode.id, 'Unknown Node') END +
                        ' to ' +
                        CASE WHEN apoc.meta.type(endNode.name) = 'LIST' THEN apoc.text.join(endNode.name, ', ') ELSE COALESCE(endNode.name, endNode.id, 'Unknown Node') END +
                        CASE
                            WHEN apoc.meta.type(relationship.type) = 'LIST' THEN ' (Type: ' + apoc.text.join(relationship.type, ', ') + ')'
                            WHEN relationship.type IS NOT NULL THEN ' (Type: ' + relationship.type + ')'
                            ELSE ''
                        END AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized MEMBER_OF GraphRetriever.")

            # 8 HAS_ENZYME_FUNCTION Relationships
            print("Configuring HAS_ENZYME_FUNCTION Relationship Neo4jVector store...")
            has_enzyme_function_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="has_enzyme_function_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship, score
                    MATCH (startNode)-[relationship]->(endNode)
                    RETURN
                        'Relationship ' + type(relationship) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to ' + COALESCE(endNode.name, endNode.id, 'Unknown Node') AS text,
                        score,
                        relationship {.*, score: score, startNodeName: COALESCE(startNode.name, startNode.id), endNodeName: COALESCE(endNode.name, endNode.id)} AS metadata
                """
            )
            print("Initialized HAS_ENZYME_FUNCTION GraphRetriever.")

            # Combine all retrievers into a single list for the MergerRetriever
            # all_individual_retrievers = [
            #     gene_vector_store.as_retriever(),
            #     ortholog_vector_store.as_retriever(),
            #     reaction_vector_store.as_retriever(),
            #     compound_vector_store.as_retriever(),
            #     pathway_vector_store.as_retriever(),
            #     pathway_entry_vector_store.as_retriever(),
            #     ec_vector_store.as_retriever(),
            #     contains_vector_store.as_retriever(),
            #     belongs_to_vector_store.as_retriever(),
            #     catalyzes_vector_store.as_retriever(),
            #     produces_vector_store.as_retriever(),
            #     encodes_vector_store.as_retriever(),
            #     substrate_of_vector_store.as_retriever(),
            #     member_of_vector_store.as_retriever(),
            #     has_enzyme_function_vector_store.as_retriever()
            # ]
            all_individual_retrievers = [
                gene_vector_store.as_retriever(),
                ortholog_vector_store.as_retriever(),
                reaction_vector_store.as_retriever(),
                compound_vector_store.as_retriever(),
                pathway_vector_store.as_retriever(),
                pathway_entry_vector_store.as_retriever(),
                ec_vector_store.as_retriever(),
            ]

            # Create the final MergerRetriever
            self.retriever = MergerRetriever(retrievers=all_individual_retrievers)

            # ... (all your individual Neo4jVector instances are created above)

            
            # Create a list to hold all the direct Neo4jVector retrievers
        
            # self.retriever = MergerRetriever(retrievers=[gene_retriever_obj, reaction_retriever_obj, pathway_retriever_obj])

            #self.retriever = MergerRetriever(retrievers=[gene_retriever_obj, reaction_retriever_obj, compound_retriever_obj, pathway_retriever_obj, pathway_entry_retriever_obj, ortholog_retriever_obj, ec_retriever_obj, belongs_to_retriever_obj, contains_retriever_obj, includes_retriever_obj, interacts_with_retriever_obj])

            print(ortholog_vector_store)
            print("Successfully initialized combined MergerRetriever.")

            for retriever in all_individual_retrievers:
                print(f"Testing {type(retriever)}")
                try:
                    docs = retriever.get_relevant_documents("test")
                    print(f"Success, got {len(docs)} docs")
                except Exception as e:
                    print(f"Error: {e}")

        except Exception as e:
            print(f"ERROR: Could not connect to Neo4jVector store(s) or retrieve index: {e}")
            print("Ensure Neo4j is running, vector indexes exist, and data/embeddings are loaded for ALL configured node types.")
            raise e

        # --- 5. RetrievalQA Chain Configuration (to answer using retrieved context) ---
        print("Initializing RetrievalQA chain...")
        QA_PROMPT = PromptTemplate.from_template(
            """You are a helpful scientific assistant. Answer the question based ONLY on the following context.
            If the answer is not found in the context, say "I don't have enough information to answer that question from the provided data."

            Context:
            {context}

            Question: {question}
            Answer:
            """
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT},
            verbose=True
        )
        print("Successfully initialized RetrievalQA chain.")

        # --- 6. Conversation Memory for Recording (still for logging) ---
        print("Initializing Conversation Memory...")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print("Conversation Memory Initialized.")


    async def process_query(self, query: str):
        response_content = ""
        source_documents = []
        try:
            result = await self.qa_chain.ainvoke({"query": query})
            response_content = result.get("result", "I couldn't find an answer based on the provided data.")
            source_documents = result.get("source_documents", [])
        except Exception as e:
            response_content = f"An error occurred while processing your query: {str(e)}"
            print(f"\nInternal Error: {str(e)}")

        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response_content)

        print(f"\nBot: {response_content}")
        if source_documents:
            print("\n--- Retrieved Source Documents ---")
            for i, doc in enumerate(source_documents[:5]):
                print(f"--- Document {i+1} ---")
                print(f"   Content: {doc.page_content[:300]}...")
                print(f"   Metadata: {doc.metadata}")
            if len(source_documents) > 5:
                print(f"...and {len(source_documents) - 5} more documents.")
            print("----------------------------------")


    async def chat_loop(self):
        print(f"\nMCP Graph Vector QA Bot Started! LLM: {self.llm.model}, Embedding: {self.embeddings.model_name}.")
        print("This bot answers questions using vector search and graph traversal in Neo4j across multiple entity types.")
        print("-" * 50)
        while True:
            query = input("\nYou: ").strip()
            if query.lower() == "quit":
                break
            try:
                await self.process_query(query)
            except Exception as e:
                print(f"\nFatal Error in chat loop: {str(e)}")
                break

if __name__ == "__main__":
    

    bot = MCP_ChatBot(llm_model_name="llama3.2:1b", embedding_model_name="NeuML/pubmedbert-base-embeddings")
    asyncio.run(bot.chat_loop())