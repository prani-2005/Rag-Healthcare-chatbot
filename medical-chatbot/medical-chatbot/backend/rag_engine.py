import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import together

load_dotenv()

class MedicalRAGEngine:
    def __init__(self):
        self.initialize_vector_store()
        self.initialize_llm()
        
    def initialize_vector_store(self):
        """Initialize connection to Pinecone"""
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "")  # Optional

        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        existing_indexes = [index.name for index in self.pinecone.list_indexes()]
        
        if self.index_name not in existing_indexes:
            raise ValueError(f"Index '{self.index_name}' not found in Pinecone.")
        
        self.index = self.pinecone.Index(self.index_name)
        print(f"Connected to Pinecone index '{self.index_name}'")
    
    def initialize_llm(self):
        """Initialize the LLM with Together API"""
        # together.api_key = os.getenv("TOGETHER_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        print(f"Initialized LLM with model: {self.model_name}")
    
    def retrieve_relevant_context(self, query, k=5):
        """Embed the query and search Pinecone for similar docs"""
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True, namespace=self.namespace)

        context_texts = []
        sources = []

        for match in results['matches']:
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            source = metadata.get("source", "Unknown")
            context_texts.append(text)
            sources.append(source)
        
        combined_context = "\n\n".join(context_texts)
        return combined_context, list(set(sources))
    
    def generate_response(self, query, context):
        """Generate a response using the LLM with retrieved context"""
        prompt = f"""You are a helpful medical assistant with access to medical literature. 
Answer the question based on the provided medical context. 
If you cannot find the answer in the context, say so clearly and provide general medical information if possible.
Do not make up information or provide medical advice that's not supported by the context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        response = together.Complete.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            stop=['QUESTION:', 'CONTEXT:']
        )
        # print(response)
        generated_text = response["choices"][0]["text"].strip() if response.get("choices") else ""

        if not generated_text:
            return "I couldn't generate a response at this time. Please try again later.", []

        return generated_text
    
    def query(self, user_query):
        """Process a user query through the RAG pipeline"""
        try:
            context, sources = self.retrieve_relevant_context(user_query)
            
            if not context.strip():
                return "I couldn't find specific information related to your query in my medical knowledge base. Please consult a healthcare professional for accurate medical advice.", []
            
            response = self.generate_response(user_query, context)
            return response, sources
        
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return "I encountered an error while processing your query. Please try again or rephrase your question.", []
