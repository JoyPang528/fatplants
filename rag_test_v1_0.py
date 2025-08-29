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


# ---------------------------
# 1) MergerRetriever (invoke)
# ---------------------------
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

# -------------------------------------
# 2) 保险阀过滤器：越界词出现则覆盖回答
# -------------------------------------

# def filter_out_of_scope_terms(self, answer: str, context: str = "") -> str:
#     blacklist = ["lipid a", "kdo2", "lps", "salicylate", "endotoxin"]
#     whitelist = ["fatty acid", "triacylglycerol", "phospholipid",
#                  "kennedy pathway", "galactolipid", "cutin", "suberin", "wax"]

#     ans_low = answer.lower()
#     ctx_low = context.lower()

#     # 黑名单触发（且 context 没说 plant） → 拒答
#     for bad in blacklist:
#         if bad in ans_low and bad not in ctx_low:
#             return "I don't have enough information to answer that question from the provided data."

#     # 白名单缺失 → 拒答
#     if not any(w in ans_low for w in whitelist):
#         return "I don't have enough information to answer that question from the provided data."

#     return answer

# ---------------------------
# 3) MCP_ChatBot
# ---------------------------
class MCP_ChatBot:
    def filter_out_of_scope_terms(self, answer: str, context: str = "") -> str:
        """
        只允许与“植物脂质生物合成”相关的回答通过：
        - 必须命中至少一个“合成通路/关键酶”白名单关键词
        - 命中黑名单（LPS/Lipid A/Kdo2/Salicylate/Endotoxin）则拦截（除非上下文确有这些词）
        - 出现过氧化/氧化降解相关词（peroxidation/oxidation/ROS/LOX/MDA）且上下文未将其定义为“合成”一部分 → 拒答
        """
        ans_low = (answer or "").lower()
        ctx_low = (context or "").lower()

        # ---- 黑名单：明确非植物脂质生物合成的路径/术语 ----
        hard_blacklist = [
            "lipid a", "kdo2", "lps", "endotoxin", "salicylate"
        ]

        # 若答案中出现黑名单，且上下文没有对应词，就拒答
        for bad in hard_blacklist:
            if bad in ans_low and bad not in ctx_low:
                return "I don't have enough information to answer that question from the provided data."

        # ---- 过氧化/氧化降解：不是生物合成 ----
        oxidation_terms = [
            "peroxidation", "lipid peroxidation", "oxidation", "reactive oxygen species",
            "ros", "lox", "lipoxygenase", "mda", "malondialdehyde"
        ]
        # 如果答案里出现“氧化/过氧化/降解”，而上下文没有把它们当作“合成/生物合成”环节，就拒答
        synthesis_markers = ["biosynth", "bio-synth", "synthes", "anabol", "formation", "pathway", "route"]
        if any(term in ans_low for term in oxidation_terms):
            if not any(marker in ctx_low for marker in synthesis_markers):
                return "I don't have enough information to answer that question from the provided data."

        # ---- 白名单：必须至少命中一项（合成主干/关键酶/典型通路） ----
        whitelist = [
            # 主干通路
            "fatty acid", "triacylglycerol", "tag", "kennedy pathway",
            "phospholipid", "galactolipid", "sulfolipid", "wax", "cutin", "suberin",
            # 关键酶/步骤
            "acc", "acetyl-coa carboxylase", "kas", "ketoacyl-acp synthase",
            "desaturase", "fad2", "fad3", "Δ9-desaturase", "delta-9 desaturase",
            "elongase", "kcs", "3-ketoacyl-coa synthase",
            "gpat", "lpaat", "pap", "dgat", "pdat", "magt", "pdct",
            "glycerol-3-phosphate", "g3p", "lysophosphatidic acid", "lpa",
            "phosphatidic acid", "pa", "diacylglycerol", "dag"
        ]

        if not any(w in ans_low for w in whitelist):
            return "I don't have enough information to answer that question from the provided data."

        # 通过所有检查，放行
        return answer


    def __init__(self, llm_model_name="gemma3:27b", embedding_model_name="NeuML/pubmedbert-base-embeddings"):
        print("MCP_ChatBot.py - Version 2025-06-27-EagerFix-AttemptFinal-Revised-v4-CorrectGraphRetriever") # Version Identifier

        # --- 1. Ollama LLM Configuration (for answering) ---
        print("Initializing LLM...")
        self.llm = ChatOllama(model=llm_model_name, 
                              base_url="http://localhost:11434", 
                              temperature=0.0, 
                              num_ctx=8192,        # or lower if your build needs it
                              num_predict=512,     # ensure not zero
                              stop=None)
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
            # --- RELATIONSHIP VECTOR STORE CONFIGURATIONS ---

            print("Configuring CONTAINS Relationship Neo4jVector store...")
            contains_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="contains_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + toString(COALESCE(startNode.name, startNode.id, 'Unknown Node')) +
                        ' to '   + toString(COALESCE(endNode.name,   endNode.id,   'Unknown Node')) +
                        CASE
                            WHEN rel.type IS NOT NULL THEN ' (Type: ' + toString(rel.type) + ')'
                            ELSE ''
                        END AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: toString(COALESCE(startNode.name, startNode.id)),
                            endNodeName:   toString(COALESCE(endNode.name,   endNode.id)),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized CONTAINS GraphRetriever.")

            print("Configuring BELONGS_TO Relationship Neo4jVector store...")
            belongs_to_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="belongs_to_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to '   + COALESCE(endNode.name,   endNode.id,   'Unknown Node') AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: COALESCE(startNode.name, startNode.id),
                            endNodeName:   COALESCE(endNode.name,   endNode.id),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized BELONGS_TO GraphRetriever.")

            print("Configuring CATALYZES Relationship Neo4jVector store...")
            catalyzes_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="catalyzes_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to '   + COALESCE(endNode.name,   endNode.id,   'Unknown Node') AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: COALESCE(startNode.name, startNode.id),
                            endNodeName:   COALESCE(endNode.name,   endNode.id),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized CATALYZES GraphRetriever.")

            print("Configuring PRODUCES Relationship Neo4jVector store...")
            produces_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="produces_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to '   + COALESCE(endNode.name,   endNode.id,   'Unknown Node') AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: COALESCE(startNode.name, startNode.id),
                            endNodeName:   COALESCE(endNode.name,   endNode.id),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized PRODUCES GraphRetriever.")

            print("Configuring ENCODES Relationship Neo4jVector store...")
            encodes_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="encodes_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to '   + COALESCE(endNode.name,   endNode.id,   'Unknown Node') AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: COALESCE(startNode.name, startNode.id),
                            endNodeName:   COALESCE(endNode.name,   endNode.id),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized ENCODES GraphRetriever.")

            print("Configuring SUBSTRATE_OF Relationship Neo4jVector store...")
            substrate_of_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="substrate_of_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to '   + COALESCE(endNode.name,   endNode.id,   'Unknown Node') AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: COALESCE(startNode.name, startNode.id),
                            endNodeName:   COALESCE(endNode.name,   endNode.id),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized SUBSTRATE_OF GraphRetriever.")

            print("Configuring MEMBER_OF Relationship Neo4jVector store...")
            member_of_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="member_of_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + toString(COALESCE(startNode.name, startNode.id, 'Unknown Node')) +
                        ' to '   + toString(COALESCE(endNode.name,   endNode.id,   'Unknown Node')) +
                        CASE
                            WHEN rel.type IS NOT NULL THEN ' (Type: ' + toString(rel.type) + ')'
                            ELSE ''
                        END AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: toString(COALESCE(startNode.name, startNode.id)),
                            endNodeName:   toString(COALESCE(endNode.name,   endNode.id)),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized MEMBER_OF GraphRetriever.")

            print("Configuring HAS_ENZYME_FUNCTION Relationship Neo4jVector store...")
            has_enzyme_function_vector_store = Neo4jVector.from_existing_relationship_index(
                embedding=self.embeddings,
                url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
                index_name="has_enzyme_function_embeddings",
                embedding_node_property="embedding",
                retrieval_query="""
                    CALL db.index.vector.queryRelationships($index, $k, $embedding)
                    YIELD relationship AS rel, score AS relScore
                    MATCH (startNode)-[rel]->(endNode)
                    RETURN
                        'Relationship ' + type(rel) +
                        ' from ' + COALESCE(startNode.name, startNode.id, 'Unknown Node') +
                        ' to '   + COALESCE(endNode.name,   endNode.id,   'Unknown Node') AS text,
                        relScore AS score,
                        rel {.*,
                            score: relScore,
                            startNodeName: COALESCE(startNode.name, startNode.id),
                            endNodeName:   COALESCE(endNode.name,   endNode.id),
                            relId: elementId(rel),
                            startId: elementId(startNode),
                            endId: elementId(endNode)} AS metadata
                """
            )
            print("Initialized HAS_ENZYME_FUNCTION GraphRetriever.")


            # Combine all retrievers into a single list for the MergerRetriever
            # Pathway-first for "pathways" style questions
            pathway_ret  = pathway_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10, "fetch_k": 60}
            )
            reaction_ret = reaction_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8, "fetch_k": 60}
            )
            compound_ret = compound_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4, "fetch_k": 36}
            )
            ec_ret = ec_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4, "fetch_k": 36}
            )
            gene_ret = gene_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2, "fetch_k": 24}
            )
            ortholog_ret = ortholog_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2, "fetch_k": 24}
            )
            

            # Relationship retrievers are often noisy for high-level questions – keep small
            contains_ret      = contains_vector_store.as_retriever(     search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            belongs_to_ret    = belongs_to_vector_store.as_retriever(   search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            catalyzes_ret     = catalyzes_vector_store.as_retriever(    search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            produces_ret      = produces_vector_store.as_retriever(     search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            encodes_ret       = encodes_vector_store.as_retriever(      search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            substrate_of_ret  = substrate_of_vector_store.as_retriever( search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            member_of_ret     = member_of_vector_store.as_retriever(    search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24})
            has_enzyme_fn_ret = has_enzyme_function_vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 2, "fetch_k": 24}
            )

            all_individual_retrievers = [
                pathway_ret, reaction_ret, compound_ret, ec_ret,
                gene_ret, ortholog_ret,
                contains_ret, belongs_to_ret, catalyzes_ret, produces_ret,
                encodes_ret, substrate_of_ret, member_of_ret, has_enzyme_fn_ret,
                pathway_entry_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4, "fetch_k": 36}),
            ]


            # Create the final MergerRetriever
            self.retriever = MergerRetriever(retrievers=all_individual_retrievers)

            
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
            """You are a domain expert in PLANT lipid metabolism. 
Answer ONLY about **plant lipid biosynthesis** and ONLY using the provided context. 
If the answer is not supported by the context, say exactly:
"I don't have enough information to answer that question from the provided data."

Strict rules:
- Scope MUST be plant lipid biosynthesis (e.g., plastid fatty acid synthesis, elongation, desaturation, Kennedy pathway to TAG, galactolipids, phospholipids, wax/cutin/suberin). 
- Ignore or flag non-plant or unrelated pathways (e.g., bacterial LPS/Lipid A/Kdo2-lipid, animal-specific sterol routes, salicylate defense signaling) unless the context explicitly links them to plant lipid biosynthesis.
- Do not infer beyond the context. Do not cite knowledge outside the context.

When answering:
1) Start with a one-sentence direct answer.
2) Then list the **key plant lipid biosynthesis pathways** found in context, each with:
   - Name
   - Subcellular compartment(s) (e.g., plastid, ER, peroxisome)
   - 1–3 hallmark enzymes/steps present in the context
3) Add a short "Why these pathways" note tying back to context.
4) Add "Out-of-scope terms (ignored)" listing any unrelated terms found in context (e.g., Lipid A, Kdo2, LPS, salicylate), if any.
5) End with "Sources" as bullet points: brief identifiers or titles from context metadata.

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
        # response_content = ""
        # source_documents = []
        # try:
        #     # result = await self.qa_chain.ainvoke({"query": query})
        #     # response_content = result.get("result", "I couldn't find an answer based on the provided data.")
        #     # source_documents = result.get("source_documents", [])
        #     result = await self.qa_chain.ainvoke({"query": query})
        #     raw_answer = result.get("result", "I couldn't find an answer based on the provided data.")
        #     source_documents = result.get("source_documents", [])

        #     context_str = "\n".join([doc.page_content for doc in source_documents])
        #     response_content = filter_out_of_scope_terms(raw_answer, context=context_str)

        """
        1) 先用 RetrievalQA 正常检索
        2) 对 source_documents 做“植物相关”过滤（白名单 + 物种前缀 + 黑名单剔除）
        3) 用过滤后的上下文重新生成答案
        4) 最后再用保险阀做黑/白名单兜底
        """
        # 关键词白/黑名单（可按需扩展）
        PLANT_WHITELIST = [
            "fatty acid", "triacylglycerol", "tag", "phospholipid",
            "kennedy pathway", "galactolipid", "sulfolipid",
            "cutin", "suberin", "wax", "desaturase", "elongase",
            "acc", "kas", "dgat", "pdat", "fad2", "fad3", "fatty-acyl"
        ]
        BLACKLIST = ["lipid a", "kdo2", "lps", "endotoxin", "salicylate"]

        # 常见植物数据库/物种前缀（用于 heuristic 判断）
        PLANT_PREFIX_HINTS = [
            "ath", "ara", "arabidopsis", "osa", "oryza", "zma", "zea", "sly",
            "solanum", "vvi", "vitis", "bra", "brassica", "cam", "camelina",
            "gly", "glycine", "med", "medicago", "pop", "populus", "ptr"
        ]

        def _is_plant_doc(doc):
            txt = (doc.page_content or "").lower()
            meta = {k.lower(): str(v).lower() for k, v in (doc.metadata or {}).items()}
            meta_str = " ".join([str(v) for v in meta.values()])

            # 黑名单命中则判为非植物
            if any(bad in txt or bad in meta_str for bad in BLACKLIST):
                return False

            # 若包含明显植物脂质关键词或常见酶名 → 认为相关
            if any(w in txt for w in PLANT_WHITELIST):
                return True

            # 若 metadata/link/id/name 中包含植物物种/前缀提示 → 认为相关
            if any(h in meta_str for h in PLANT_PREFIX_HINTS):
                return True

            # 否则保守地视为不相关
            return False

        try:
            # 1) 先跑原始链
            result = await self.qa_chain.ainvoke({"query": query})
            source_documents = result.get("source_documents", [])
            raw_answer = result.get("result", "I couldn't find an answer based on the provided data.")

            # 2) 过滤 source_documents
            filtered_docs = [d for d in source_documents if _is_plant_doc(d)]

            # 如果过滤后为空，就退而求其次：只去掉明显黑名单的
            if not filtered_docs:
                filtered_docs = [
                    d for d in source_documents
                    if not any(b in (d.page_content or "").lower() for b in BLACKLIST)
                ]

            # 3) 用过滤后的上下文重新生成答案（同样的 QA_PROMPT 结构）
            if filtered_docs:
                filtered_context = "\n\n".join(d.page_content for d in filtered_docs if d.page_content)
                prompt_text = (
                    "You are a domain expert in PLANT lipid metabolism.\n"
                    "Answer ONLY about **plant lipid biosynthesis** and ONLY using the provided context.\n"
                    "If the answer is not supported by the context, say exactly:\n"
                    "\"I don't have enough information to answer that question from the provided data.\"\n\n"
                    "Strict rules:\n"
                    "- Scope MUST be plant lipid biosynthesis (e.g., plastid fatty acid synthesis, elongation, desaturation, Kennedy pathway to TAG, galactolipids, phospholipids, wax/cutin/suberin).\n"
                    "- Ignore or flag non-plant or unrelated pathways (e.g., bacterial LPS/Lipid A/Kdo2-lipid, animal-specific sterol routes, salicylate defense signaling) unless the context explicitly links them to plant lipid biosynthesis.\n"
                    "- Do not infer beyond the context. Do not cite knowledge outside the context.\n\n"
                    "When answering:\n"
                    "1) Start with a one-sentence direct answer.\n"
                    "2) Then list the **key plant lipid biosynthesis pathways** found in context, each with:\n"
                    "   - Name\n"
                    "   - Subcellular compartment(s) (e.g., plastid, ER, peroxisome)\n"
                    "   - 1–3 hallmark enzymes/steps present in the context\n"
                    "3) Add a short \"Why these pathways\" note tying back to context.\n"
                    "4) Add \"Out-of-scope terms (ignored)\" listing any unrelated terms found in context (e.g., Lipid A, Kdo2, LPS, salicylate), if any.\n"
                    "5) End with \"Sources\" as bullet points: brief identifiers or titles from context metadata.\n\n"
                    f"Context:\n{filtered_context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer:\n"
                )
                regenerated = self.llm.invoke(prompt_text)
                regenerated_answer = regenerated.content if hasattr(regenerated, "content") else str(regenerated)
            else:
                regenerated_answer = raw_answer  # 没有可用上下文就保留原答案

            # 4) 最终保险阀（黑/白名单 + 与上下文的一致性）
            final_context = "\n".join([d.page_content for d in filtered_docs]) if filtered_docs else \
                            "\n".join([d.page_content for d in source_documents])
            response_content = self.filter_out_of_scope_terms(regenerated_answer, context=final_context)

            
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