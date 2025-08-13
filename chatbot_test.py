import os
import requests
import json
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory # Corrected import for ChatMessageHistory
from typing import AsyncIterable, Dict, Any, List, Union

# --- Configuration (from Environment Variables) ---
# Ensure these environment variables are set before running:
# export OLLAMA_BASE_URL="http://localhost:11434" # Default Ollama URL
# export CITATION_SERVICE_URL="http://localhost:5001/retrieve-citations"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CITATION_SERVICE_URL = os.getenv("CITATION_SERVICE_URL", "http://127.0.0.1:5001/retrieve-citations")

# --- Model Configuration ---
# Set the default model to your local Gemma model
DEFAULT_LLM_MODEL = "llama3.2:1b" # Changed to a smaller model (e.g., 1.3B) for better compatibility

class ChatbotService:
    def __init__(self):
        self.logger = print # Simple logger for demonstration - MOVED TO TOP
        self.llm_client: ChatOllama = self._initialize_llm_client()
        # Initialize LangChain memory
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        # MODIFIED SYSTEM PROMPT for main answer generation
        self.MAIN_ANSWER_SYSTEM_PROMPT = """Answer the following biomedical question in a very specific manner:
1. Content Requirements:
- Provide only the names of the genes, pathways, or gene-protein interactions when the question specifically asks for them.
- Do not include any extra explanations or additional information unless explicitly requested in the query.
- Highlight only the main keywords, genes, pathways, or their interactions when asked.

2. Additional Notes:
- Ensure that the answer is as precise and accurate as possible.
- Do not include any citations or web scraping results in your answer. These will be provided separately.

Please strictly follow these guidelines in your responses."""

    def _initialize_llm_client(self) -> ChatOllama:
        """Initializes the LLM client to connect to the local Ollama server using ChatOllama."""
        self.logger(f"Initializing LLM client for local Ollama at: {OLLAMA_BASE_URL} using ChatOllama. Model: {DEFAULT_LLM_MODEL}")
        return ChatOllama(
            model=DEFAULT_LLM_MODEL, # Model is set here once for the client instance
            base_url=OLLAMA_BASE_URL,
            temperature=0.1 # Using temperature here for ChatOllama
        )

    async def _create_static_stream(self, message: str) -> AsyncIterable[Dict[str, Any]]:
        """Helper to create an async iterable for static messages (mimicking OpenAI stream format)."""
        yield {"choices": [{"delta": {"content": message}}]}

    async def _create_error_stream(self, error_message: str) -> AsyncIterable[Dict[str, Any]]:
        """Helper to create an async iterable for error messages (mimicking OpenAI stream format)."""
        yield {"choices": [{"delta": {"content": f"Error: {error_message}"}}]}

    async def _create_combined_stream(
        self,
        citations: List[str],
        llm_stream: AsyncIterable[str] # LangChain stream yields strings directly
    ) -> AsyncIterable[Dict[str, Any]]:
        """Helper to combine citations and LLM stream into a single async iterable
           (outputting in a format similar to OpenAI stream for consistency)."""
        if citations:
            yield {"choices": [{"delta": {"content": "\n\n--- Relevant Citations ---"}}]}
            for citation in citations:
                yield {"choices": [{"delta": {"content": "\n" + citation + "\n"}}]}
            yield {"choices": [{"delta": {"content": "\n\n--- Answer ---"}}]}

        async for chunk in llm_stream:
            # LangChain's stream yields strings, wrap it in the expected dict format
            yield {"choices": [{"delta": {"content": chunk}}]}

    async def generate_response_stream(self, query: str) -> AsyncIterable[Dict[str, Any]]: # Removed prev_messages parameter
        """
        Generates a streaming response from the chatbot, including query classification,
        keyword extraction, citation retrieval, and LLM answer generation.
        """
        # Add user message to memory
        self.memory.chat_memory.add_user_message(query)

        # --- Step 1: Query Classification ---
        classification_prompt_content = f"""You are a biologist.
        Is the following query related to plant fat-related genes, proteins, and metabolism?
        Respond with 'YES' or 'NO', followed by a brief reason (1–2 sentences max).
        Query: "{query}" """

        try:
            # ChatOllama uses .invoke() for non-streaming calls
            classification_response = await self.llm_client.ainvoke( # Removed config={"model": DEFAULT_LLM_MODEL}
                [HumanMessage(content=classification_prompt_content)]
            )
            classification_text = classification_response.content.strip().upper()
            self.logger(f"Classification LLM Response: {classification_text}")

            if not classification_text.startswith('YES'):
                irrelevant_message = "The query is not about plant fat-related genes, proteins, and metabolism. Please ask a question related to this topic."
                # Record the irrelevant message in memory as AI's response
                self.memory.chat_memory.add_ai_message(irrelevant_message)
                return self._create_static_stream(irrelevant_message)

        except Exception as e:
            self.logger(f"Error during query classification: {e}")
            error_msg = f"Error classifying query: {e}"
            self.memory.chat_memory.add_ai_message(error_msg) # Record error in memory
            return self._create_error_stream(error_msg)

        # --- If relevant, proceed with citation retrieval using original query ---
        # Removed Step 2 (LLM Keyword Extraction) as it's now handled by citation_retrieval_module_python
        
        # citations: List[str] = []
        # try:
        #     # --- Step 3: Call Python Citation Retrieval Microservice ---
        #     # Send the ORIGINAL user query to the citation service for internal entity extraction
        #     query_for_citation = query 
        #     self.logger(f"Calling citation service with query: '{query_for_citation}'")
        #     citation_response = requests.post(
        #         CITATION_SERVICE_URL,
        #         json={"query": query_for_citation, "num_citations": 3}
        #     )
        #     citation_response.raise_for_status() # Raise an exception for HTTP errors
        #     citations = citation_response.json().get("citations", [])
        #     self.logger(f"Retrieved citations: {citations}")
        # except requests.exceptions.RequestException as e:
        #     self.logger(f"Error fetching citations from Python service: {e}")
        #     citations = ["Not able to scrape citations for this question."]
        # except json.JSONDecodeError as e:
        #     self.logger(f"Error decoding JSON from citation service: {e}. Response: {citation_response.text}")
        #     citations = ["Error processing citation data."]

        citations: List[str] = []
        try:
            # --- Step 3: Call Python Citation Retrieval Microservice ---
            # Send the ORIGINAL user query to the citation service for internal entity extraction
            query_for_citation = query 
            self.logger(f"Calling citation service with query: '{query_for_citation}'")
            citation_response = requests.post(
                CITATION_SERVICE_URL,
                json={"query": query_for_citation, "num_citations": 3}
            )
            citation_response.raise_for_status() # Raise an exception for HTTP errors
            citations = citation_response.json().get("citations", [])
            # Corrected logging of citations: join them with newlines for readability
            citation_text = '\n'.join(citations)
            self.logger(f"Retrieved citations:\n{citation_text}")

            #self.logger(f"Retrieved citations:\n{'\\n'.join(citations)}") # FIX: Corrected f-string syntax
        except requests.exceptions.RequestException as e:
            self.logger(f"Error fetching citations from Python service: {e}")
            # If citation retrieval fails, return an error stream immediately
            error_msg = f"Failed to retrieve citations: {e}. Please ensure the citation service is running."
            self.memory.chat_memory.add_ai_message(error_msg)
            return self._create_error_stream(error_msg)
        except json.JSONDecodeError as e:
            self.logger(f"Error decoding JSON from citation service: {e}. Response: {citation_response.text}")
            error_msg = f"Error processing citation data from service: {e}."
            self.memory.chat_memory.add_ai_message(error_msg)
            return self._create_error_stream(error_msg)


        # --- Step 4: Main Answer Generation ---
        # LangChain messages: Get all messages from memory
        messages_for_llm = self.memory.chat_memory.messages.copy() # Use .copy() to avoid modifying original list during this call

        # Add the current user query if it's not already the last message (it should be)
        if not messages_for_llm or messages_for_llm[-1].content != query:
             messages_for_llm.append(HumanMessage(content=query)) # Should be added by add_user_message already

        # Prepend the system prompt at the very beginning of the conversation history
        messages_for_llm.insert(0, SystemMessage(content=self.MAIN_ANSWER_SYSTEM_PROMPT))

        # Optional: Add citations to the LLM's context as a SystemMessage
        if citations and citations[0] != "Not able to scrape citations for this question.":
            # Corrected f-string syntax
            citations_content = '\n\n'.join(citations)
            messages_for_llm.append(
                SystemMessage(content=f"Here are some relevant citations that you can refer to in your answer if appropriate, but do not format them yourself or include links: \n{citations_content}")
            )

        llm_stream_raw: AsyncIterable[str] # LangChain's .astream() yields strings
        try:
            llm_stream_raw = self.llm_client.astream( # Removed config={"model": DEFAULT_LLM_MODEL}
                messages_for_llm
            )

        except Exception as e:
            self.logger(f"Error generating main LLM response: {e}")
            error_msg = f"Error generating answer: {e}"
            self.memory.chat_memory.add_ai_message(error_msg) # Record error in memory
            return self._create_error_stream(error_msg)

        # --- Step 5: Combine LLM stream with citations and return ---
        # Capture the full response content to add to memory
        full_response_content = ""
        async def stream_and_capture():
            nonlocal full_response_content # Allow modification of outer variable
            async for message_chunk in llm_stream_raw: # Iterate directly over message_chunk objects
                # Access the content attribute of the message_chunk
                content = message_chunk.content
                # LangChain's astream yields strings directly, so we need to wrap them
                chunk_dict = {"choices": [{"delta": {"content": content}}]}
                full_response_content += content
                yield chunk_dict
        
        # This part of the code is tricky with async generators and memory.
        # The AI message should be added to memory *after* the stream is fully consumed.
        # So, we'll return a new generator that adds to memory upon completion.
        async def final_response_generator():
            nonlocal full_response_content
            # Consume the stream from stream_and_capture
            async for chunk in stream_and_capture():
                yield chunk
            # Once the stream is fully consumed, add the complete AI message to memory
            self.memory.chat_memory.add_ai_message(full_response_content)

        return final_response_generator()

# --- Example Usage ---
async def main():
    # To run this example, ensure:
    # 1. Your Python Flask citation_api.py is running on port 5001.
    # 2. You have Ollama running and the 'gemma:2b' model pulled:
    #    - Download Ollama: https://ollama.com/download
    #    - Pull the model: ollama pull gemma:2b
    # 3. You have set OLLAMA_BASE_URL environment variable (if not default http://localhost:11434).
    # 4. You have set CITATION_SERVICE_URL environment variable.
    # Then, run this script: python your_chatbot_script_name.py

    chatbot = ChatbotService()
    
    print("Welcome to the Plant Fat Metabolism Chatbot!")
    print("Ask questions about plant fat-related genes, proteins, and metabolism.")
    print("Type 'exit' or 'quit' to end the chat.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        print("Chatbot: ", end="", flush=True)
        # The full_response_content is now captured and added to memory internally by the service
        try:
            # Await the coroutine returned by generate_response_stream
            async for chunk_dict in await chatbot.generate_response_stream(user_query):
                content = chunk_dict["choices"][0]["delta"].get("content", "")
                # print(content, end="", flush=True)
            print() # Newline after streaming
            
        except Exception as e:
            print(f"An error occurred: {e}")
            # Error messages are already added to memory by helper functions

        # You can inspect the memory after each turn for debugging
        # print("\n--- Current Chat History (from memory) ---")
        # for msg in chatbot.memory.chat_memory.messages:
        #     print(f"{msg.type}: {msg.content}")
        # print("--------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
