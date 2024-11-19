import google.generativeai as genai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

class AIQueryProcessor:
    def __init__(self, google_api_key, pinecone_api_key, index_name):
        # Initialize Google Gemini LLM
        self.google_api_key = google_api_key
        genai.configure(api_key=self.google_api_key)
        self.llm = genai.GenerativeModel(model_name="gemini-pro")
        
        # Initialize Hugging Face models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.slm_a_name = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
        self.slm_b_name = 'microsoft/BioGPT-Large'
        self.slm_a, self.slm_a_tokenizer = self._init_model(self.slm_a_name)
        self.slm_b, self.slm_b_tokenizer = self._init_model(self.slm_b_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        indexes = []
        for i in pc.list_indexes():
            indexes.append(i['name'])
            
        if index_name not in indexes:
            pc.create_index(index_name, dimension=384)  # Adjust dimension based on embedding size
        self.index = pc.Index(index_name)

    def _init_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def _generate_embeddings(self, text):
        return self.embedding_model.encode(text).tolist()

    def process_query(self, query):
        # Generate query embedding
        query_embedding = self._generate_embeddings(query)
        
        # Search Pinecone for similar queries
        results = self.index.query(vector=query_embedding, top_k=1, include_metadata=True)
        if results["matches"] and results["matches"][0]["score"] > 0.8:  # Adjust threshold
            print("Cache hit!")
            return results["matches"][0]["metadata"]["response"]
        
        print("Cache miss! Generating new response.")
        
        # Process the query through LLMs
        query_a, query_b = self._analyze_query(query)
        response_a = self._get_response(self.slm_a, self.slm_a_tokenizer, query_a)
        response_b = self._get_response(self.slm_b, self.slm_b_tokenizer, query_b)
        
        combined_response = self._combine_responses(query, response_a, response_b)
        
        # Store the query and response in Pinecone
        self.index.upsert([
            {
                "id": query,  # Use the query text as the ID
                "values": query_embedding,
                "metadata": {"query": query, "response": combined_response}
            }
        ])
        
        return combined_response

    def _analyze_query(self, query):
        
        prompt = f"""Split this query into computer science and biology parts:a
        Query: {query}

        Return exactly in this format:
        CS: [CS/ML part of the query]
        BIO: [Biology part of the query]"""
        
        response = self.llm.generate_content(prompt).text.strip()

        # Add error handling for response parsing
        if 'CS:' not in response or 'BIO:' not in response:
            return query, query

        part_a = response[response.find('CS:'):response.find('BIO:')].replace('CS:', '').strip()
        part_b = response[response.find('BIO:'):].replace('BIO:', '').strip()
        
        return part_a, part_b

    def _get_response(self, model, tokenizer, query):
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _combine_responses(self, original_query, response_cs, response_bio):
        prompt = f"""
        Original Query: {original_query}
        CS Response: {response_cs}
        BIO Response: {response_bio}
        Combine these into a unified response.
        """
        return self.llm.generate_content(prompt).text.strip()


# Example Usage
if __name__ == "__main__":
    
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    INDEX_NAME = os.getenv('INDEX_NAME')

    processor = AIQueryProcessor(
        google_api_key=GOOGLE_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME
    )

#     query = "How does quantum computing affect biolical?"
#     query = "What role does reinforcement learning play in optimizing protein folding simulations?"
    query = "How can reinforcement learning contribute to improving the efficiency of protein folding prediction models?"
    response = processor.process_query(query)
    print("Response:", response)