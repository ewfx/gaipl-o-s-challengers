import os
import logging
import re
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_system")

class NetworkRAGSystem:
    def __init__(self, knowledge_base_path="/content/agentic_rag-mcp_system/network_knowledge_base.txt"):
        self.doc_collections = {}
        self.vector_stores = {}
        self.llm = None
        self.initialized = False
        self.knowledge_base_path = knowledge_base_path

    def initialize(self):
        """Initialize the RAG system with better error handling"""
        if self.initialized:
            return True

        logger.info("Initializing RAG system...")

        # Test section mapping
        test_sections = ["TROUBLESHOOTING", "OBSERVABILITY", "DEVICE SEARCH", "KNOWLEDGE BASE", "INCIDENT", "DEVICE INVENTORY"]
        for section in test_sections:
            mapped = self._map_section_to_agent(section)
            logger.info(f"Test mapping: '{section}' → '{mapped}'")

        # Sequential initialization with verification
        if not self._load_documents():
            logger.error("Document loading failed, initialization incomplete")
            return False

        if not self._create_vector_stores():
            logger.error("Vector store creation failed, initialization incomplete")
            return False

        if not self._init_llm():
            logger.error("LLM initialization failed, initialization incomplete")
            return False

        self.initialized = True
        logger.info("RAG system successfully initialized")
        return True

    def _load_documents(self):
        """Load documents from file with enhanced validation and error reporting"""
        logger.info(f"Loading documents from {self.knowledge_base_path}...")

        # Verify file exists
        kb_path = Path(self.knowledge_base_path)
        if not kb_path.exists():
            logger.error(f"Knowledge base file not found at {kb_path.absolute()}")
            return False

        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                logger.error("Knowledge base file is empty")
                return False

            logger.info(f"Successfully read knowledge base file, size: {len(content)} bytes")

            # Print raw section names for debugging
            section_matches = re.findall(r'# (\w+(?:\s+\w+)*) DOCUMENTS #', content)
            logger.info(f"Found raw section names: {section_matches}")

            # Split content into document collections by section
            section_pattern = r'# (\w+(?:\s+\w+)*) DOCUMENTS #'
            sections = re.split(section_pattern, content)

            if len(sections) <= 1:
                logger.error("Could not parse any sections from knowledge base")
                return False

            logger.info(f"Found {(len(sections)-1)//2} sections in knowledge base")

            # Process each section (odd indices are section names, even indices are content)
            for i in range(1, len(sections), 2):
                if i+1 < len(sections):
                    section_name = sections[i].strip()
                    section_content = sections[i+1].strip()
                    logger.info(f"Processing section: {section_name}, content length: {len(section_content)}")

                    # Extract documents (lines starting with "DOCUMENT X:")
                    docs = []
                    for doc in re.split(r'DOCUMENT \w+:', section_content):
                        if doc.strip():
                            docs.append(doc.strip())

                    if not docs:
                        logger.warning(f"No documents extracted from section {section_name}")
                        continue

                    # Map section names to agent types
                    agent_type = self._map_section_to_agent(section_name)
                    logger.info(f"Mapped section '{section_name}' to agent type '{agent_type}'")

                    # Store in document collections
                    self.doc_collections[agent_type] = docs
                    logger.info(f"Loaded {len(docs)} documents for {agent_type} agent")

                    # Print first few chars of first doc for verification
                    if docs:
                        logger.info(f"Sample doc for {agent_type}: {docs[0][:100]}...")

            # Verify we loaded at least some documents
            if not self.doc_collections:
                logger.error("No documents were successfully loaded")
                return False

            logger.info(f"Successfully loaded documents for {list(self.doc_collections.keys())}")
            return True

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _map_section_to_agent(self, section_name):
        """Map section names to agent types with improved matching"""
        # Normalize section name
        section_name = section_name.lower().strip()

        # Direct mappings for commonly used section names
        direct_mappings = {
            "device inventory": "device_inventory",
            "device search": "device_search",
            "troubleshooting": "troubleshooting",
            "observability": "observability",
            "knowledge base": "knowledge_base",
            "incident resolution": "incident_resolution",
            "incident": "incident_resolution"
        }

        # Check for exact match first
        if section_name in direct_mappings:
            return direct_mappings[section_name]

        # Keyword-based mapping as fallback
        mappings = {
            "troubleshooting": ["troubleshoot", "trouble", "issue", "problem", "error", "diagnos"],
            "observability": ["observ", "monitor", "metric", "alert", "threshold", "capac", "trend"],
            "device_search": ["device search", "topolog", "network", "infrastructure", "dependencies"],
            "device_inventory": ["inventory", "device inventory", "equipment", "assets", "ci"],
            "knowledge_base": ["knowledge", "protocol", "secur", "best", "practice", "reference", "guide"],
            "incident_resolution": ["incident", "sever", "resolut", "response", "communication"]
        }

        # Check each agent type's keyword list
        for agent_type, keywords in mappings.items():
            for keyword in keywords:
                if keyword in section_name:
                    logger.info(f"Mapped section '{section_name}' to agent '{agent_type}' via keyword '{keyword}'")
                    return agent_type

        # Default mapping
        logger.warning(f"No mapping found for section '{section_name}', using default mapping")
        if "device" in section_name:
            return "device_search"
        return "knowledge_base"  # Better default than "general"

    def _create_vector_stores(self):
        """Create vector stores with validation and diagnostics"""
        logger.info("Creating vector stores...")

        if not self.doc_collections:
            logger.error("No document collections available to index")
            return False

        try:
            # Initialize embeddings model with validation
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                # Quick test to verify embeddings work
                test_embedding = embeddings.embed_query("test")
                if not test_embedding or len(test_embedding) == 0:
                    raise ValueError("Embedding model returned empty embeddings")
                logger.info(f"Embeddings validated, dimension: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings model: {e}")
                return False

            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            # Process each document collection
            successful_stores = 0
            for agent_type, docs in self.doc_collections.items():
                texts = []
                metadatas = []

                # Split documents into chunks
                for i, doc in enumerate(docs):
                    chunks = text_splitter.split_text(doc)
                    logger.info(f"Split document {i+1} for {agent_type} into {len(chunks)} chunks")
                    for j, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadatas.append({
                            "source": f"Document {i+1} for {agent_type}",
                            "agent_type": agent_type,
                            "chunk_id": j
                        })

                # Create vector store if we have documents
                if texts:
                    logger.info(f"Creating vector store for {agent_type} with {len(texts)} chunks")
                    try:
                        vector_store = Chroma.from_texts(
                            texts=texts,
                            embedding=embeddings,
                            metadatas=metadatas
                        )

                        # Validate the vector store with a simple query
                        test_results = vector_store.similarity_search(f"test query for {agent_type}", k=1)
                        if len(test_results) > 0:
                            logger.info(f"Vector store for {agent_type} validated")
                            self.vector_stores[agent_type] = vector_store
                            successful_stores += 1
                        else:
                            logger.error(f"Vector store for {agent_type} failed validation check")
                    except Exception as e:
                        logger.error(f"Error creating vector store for {agent_type}: {e}")
                else:
                    logger.warning(f"No text chunks generated for {agent_type}")

            # Verify at least some vector stores were created
            if successful_stores == 0:
                logger.error("No vector stores were successfully created")
                return False

            logger.info(f"Successfully created {successful_stores} vector stores: {list(self.vector_stores.keys())}")
            return True

        except Exception as e:
            logger.error(f"Error in vector store creation: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _init_llm(self):
        """Initialize the language model with proper token limits"""
        logger.info("Loading language model...")

        try:
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto"
            )

            # Fix: Use max_new_tokens instead of max_length
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,  # Allow generating up to 512 new tokens
                do_sample=True,      # Enable sampling
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)

            # Validate the LLM with a simple query
            test_response = self.llm.invoke("Test query to verify model is working.")
            if not test_response:
                logger.error("LLM validation failed - empty response")
                return False

            logger.info("LLM initialized and validated")
            return True

        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Create a simple mock LLM for testing
            from langchain.llms.fake import FakeListLLM
            responses = [
                "This is a mock response because the real LLM could not be initialized. Please check the logs for details."
            ]
            self.llm = FakeListLLM(responses=responses)
            logger.warning("Using mock LLM due to initialization error")
            return False

    def _query_all_stores(self, query_text, k=3):
        """Query all available vector stores when the specific one isn't found"""
        logger.info(f"Performing cross-store query: {query_text}")

        all_docs = []
        all_sources = []

        # Sort stores by potential relevance (device queries should check device-related stores first)
        prioritized_stores = sorted(
            self.vector_stores.items(),
            key=lambda x: 1 if "device" in x[0] else 2  # Prioritize device-related stores
        )

        for store_name, vector_store in prioritized_stores:
            try:
                logger.info(f"Querying '{store_name}' store")
                docs = vector_store.similarity_search(query_text, k=k)
                if docs:
                    logger.info(f"Found {len(docs)} docs in '{store_name}' store")
                    all_docs.extend(docs)
                    all_sources.extend([doc.metadata.get("source", f"Unknown from {store_name}") for doc in docs])
            except Exception as e:
                logger.warning(f"Error querying {store_name} store: {e}")

        # Sort by relevance (simplistic approach)
        return all_docs[:k*2], all_sources[:k*2]  # Return more than k to allow filtering

    def query(self, agent_type, query_text, k=3, fallback_to_general=True):
        """Query the RAG system with enhanced retrieval and cross-store fallback"""
        if not self.initialized:
            logger.info("System not initialized, initializing now...")
            if not self.initialize():
                return {
                    "response": "The RAG system failed to initialize properly. Please check the logs.",
                    "sources": [],
                    "retrieved_content": []
                }

        # Add diagnostic logging
        logger.info(f"Available vector stores: {list(self.vector_stores.keys())}")
        logger.info(f"Available document collections: {list(self.doc_collections.keys())}")

        # Handle case where query_text might be a dictionary
        if isinstance(query_text, dict):
            if "description" in query_text:
                query_text = query_text["description"]
            else:
                query_text = str(query_text)

        # Preprocess query
        query_text = query_text.strip().lower()

        logger.info(f"RAG query for {agent_type}: {query_text}")

        # Get the vector store for this agent
        vector_store = self.vector_stores.get(agent_type)
        docs = []
        sources = []

        if not vector_store:
            logger.warning(f"No vector store available for {agent_type}")

            # Try cross-store searching first
            logger.info("Attempting cross-store search")
            docs, sources = self._query_all_stores(query_text, k=k)

            if not docs and fallback_to_general:
                # If cross-store search fails, try the traditional fallbacks
                if "knowledge_base" in self.vector_stores:
                    logger.info(f"Falling back to 'knowledge_base' vector store")
                    agent_type = "knowledge_base"
                    vector_store = self.vector_stores["knowledge_base"]
                elif self.vector_stores:
                    # Last resort - use any available store
                    fallback_agent = next(iter(self.vector_stores.keys()))
                    logger.info(f"Falling back to '{fallback_agent}' vector store")
                    agent_type = fallback_agent
                    vector_store = self.vector_stores[fallback_agent]
                else:
                    logger.error("No vector stores available")
                    return {
                        "response": f"I don't have specific knowledge for this query type: {agent_type}.",
                        "sources": [],
                        "retrieved_content": []
                    }

        # If we got docs from cross-store search, use those directly
        if not docs and vector_store:
            try:
                # Retrieve relevant documents from the specific store
                retrieval_k = k * 2  # Retrieve more docs than needed for filtering
                logger.info(f"Performing similarity search for {agent_type}, k={retrieval_k}")
                docs = vector_store.similarity_search(query_text, k=retrieval_k)
                logger.info(f"Retrieved {len(docs)} documents")

                # Extract sources
                sources = [doc.metadata.get("source", "Unknown") for doc in docs]

            except Exception as e:
                logger.error(f"Error in similarity search: {e}")
                docs = []
                sources = []

        if not docs:
            logger.warning(f"No documents retrieved for query: {query_text}")
            return {
                "response": "I couldn't find specific information to answer your query. Please try rephrasing your question.",
                "sources": [],
                "retrieved_content": []
            }

        # Extract content
        retrieved_contents = [doc.page_content for doc in docs]

        # Filter to most relevant top k
        retrieved_contents = retrieved_contents[:k]
        sources = sources[:k]

        # Log sources for debugging
        for i, (src, content) in enumerate(zip(sources, retrieved_contents)):
            logger.info(f"Retrieved doc {i+1}: {src}")
            logger.info(f"Content preview: {content[:100]}...")

        # Format retrieved content
        context = "\n\n".join([f"Content from {src}:\n{content}" for src, content in zip(sources, retrieved_contents)])

        # Create agent-specific prompts
        if agent_type == "troubleshooting":
            system_prompt = "You are a network troubleshooting expert. Use the retrieved information to diagnose the problem."
            prompt_template = """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
I need help troubleshooting this network issue:
{query}

Here is relevant information from our knowledge base:
{context}

Based on this information, please provide:
1. An analysis of the likely root cause
2. Possible contributing factors
3. Recommended troubleshooting steps
4. Priority assessment
<|im_end|>
<|im_start|>assistant
"""
        elif agent_type == "device_search" or agent_type == "device_inventory":
            system_prompt = "You are a network topology expert. Help identify device information and relationships."
            prompt_template = """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
I need information about network devices:
{query}

Here is relevant information about our network devices:
{context}

Based on this information, please provide:
1. Detailed device information that matches the query
2. The importance of these devices in the network
3. Any relationships or dependencies with other devices
<|im_end|>
<|im_start|>assistant
"""
        else:
            # Generic prompt for other agent types
            system_prompt = "You are a network expert. Provide information based on the retrieved knowledge."
            prompt_template = """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{query}

Here is relevant information:
{context}

Please provide a helpful response based on this information.
<|im_end|>
<|im_start|>assistant
"""

        # Format the prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(
            system_prompt=system_prompt,
            query=query_text,
            context=context
        )

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)

            # Return results
            return {
                "response": response,
                "sources": sources,
                "retrieved_content": retrieved_contents
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                "response": f"I retrieved relevant information but encountered an error when generating a response: {str(e)}",
                "sources": sources,
                "retrieved_content": retrieved_contents
            }

# Function used in gradio_app.py to initialize the LLM for LangGraph
def init_llm_for_langgraph():
    """Loading TinyLlama model for LangGraph with proper token limits"""
    logger.info("Loading TinyLlama model for LangGraph...")

    try:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )

        # Fix: Use max_new_tokens instead of max_length
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Allow generating up to 512 new tokens
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")

        # Create a simple mock LLM for testing
        from langchain.llms.fake import FakeListLLM
        responses = [
            "This is a mock response for testing purposes. The real LLM could not be initialized."
        ]
        logger.warning("Using mock LLM due to initialization error")
        return FakeListLLM(responses=responses)
