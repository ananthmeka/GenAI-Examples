import psutil
import platform
import torch
import os
from llama_cpp import Llama
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from tqdm import tqdm
import json
import pandas as pd


class SystemOptimizer:
    def __init__(self):
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """Gather system information"""
        return {
            'cpu_cores': psutil.cpu_count(logical=False),
            'total_threads': psutil.cpu_count(logical=True),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'available_ram_gb': psutil.virtual_memory().available / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
            'os': platform.system(),
            'architecture': platform.machine()
        }
    
    def get_optimal_settings(self):
        """Calculate optimal settings based on system resources"""
        info = self.system_info
        
        # Calculate optimal number of threads
        optimal_threads = max(1, info['cpu_cores'] - 1)  # Leave one core free
        
        # Calculate optimal batch size
        if info['ram_gb'] < 8:
            batch_size = 2
        elif info['ram_gb'] < 16:
            batch_size = 4
        else:
            batch_size = 8
            
        # Calculate optimal context size
        available_ram = info['available_ram_gb']
        if available_ram < 4:
            context_size = 1024
        elif available_ram < 8:
            context_size = 2048
        else:
            context_size = 4096
            
        # Recommend model based on available resources
        if info['ram_gb'] < 8:
            recommended_model = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"
        elif info['ram_gb'] < 16:
            recommended_model = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        else:
            recommended_model = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
            
        return {
            'n_threads': optimal_threads,
            'batch_size': batch_size,
            'context_size': context_size,
            'recommended_model': recommended_model,
            'use_gpu': info['gpu_available'] and info['gpu_memory'] > 4,
            'chunk_size': min(1000, context_size // 2),
            'chunk_overlap': min(100, context_size // 10)
        }
    
    def print_recommendations(self):
        """Print system info and recommendations"""
        info = self.system_info
        settings = self.get_optimal_settings()
        
        print("\n=== System Information ===")
        print(f"CPU Cores: {info['cpu_cores']} (Total threads: {info['total_threads']})")
        print(f"RAM: {info['ram_gb']:.1f} GB (Available: {info['available_ram_gb']:.1f} GB)")
        print(f"GPU: {'Available - ' + info['gpu_name'] if info['gpu_available'] else 'Not available'}")
        print(f"Operating System: {info['os']} ({info['architecture']})")
        
        print("\n=== Recommended Settings ===")
        print(f"Number of Threads: {settings['n_threads']}")
        print(f"Batch Size: {settings['batch_size']}")
        print(f"Context Size: {settings['context_size']}")
        print(f"Chunk Size: {settings['chunk_size']}")
        print(f"Recommended Model: {settings['recommended_model']}")
        print(f"GPU Usage: {'Enabled' if settings['use_gpu'] else 'Disabled'}")
        
        print("\n=== Additional Recommendations ===")
        if info['ram_gb'] < 8:
            print("⚠️ Low RAM detected. Consider closing other applications while running.")
        if info['available_ram_gb'] < 2:
            print("⚠️ Very low available RAM. Processing may be slow.")
        if info['cpu_cores'] < 4:
            print("⚠️ Limited CPU cores. Processing may take longer.")

class LaptopFriendlyQAGenerator:
    def __init__(self, model_path: str, n_threads: int = None, context_size: int = None, batch_size: int = None):
        # Get optimal settings if not provided
        if any(param is None for param in [n_threads, context_size, batch_size]):
            optimizer = SystemOptimizer()
            settings = optimizer.get_optimal_settings()
            n_threads = n_threads or settings['n_threads']
            context_size = context_size or settings['context_size']
            batch_size = batch_size or settings['batch_size']
        
        print(f"Initializing with {n_threads} threads and context size of {context_size}")
        
        # Initialize the model
        self.model = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_threads=n_threads,
            n_batch=batch_size
        )
        
        # Initialize text splitter with optimal chunk size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min(1000, context_size // 2),
            chunk_overlap=min(100, context_size // 10)
        )
    
    def generate_qa_pairs(self, text: str, max_pairs: int = 2) -> list[dict[str, str]]:
        """Generate Q&A pairs with memory-efficient processing."""
        prompt = f"""Based on this text, generate {max_pairs} question-answer pairs.
        Format: JSON array with 'question' and 'answer' keys.
        
        Text: {text}
        
        Q&A pairs:"""
        
        try:
            response = self.model(
                prompt,
                max_tokens=512,
                temperature=0.7,
                echo=False
            )
            
            # Extract JSON from response
            response_text = response['choices'][0]['text']
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            return []
            
        except Exception as e:
            print(f"Error generating Q&A pairs: {e}")
            return []
    
    def load_document(self, file_path: str) -> list[str]:
        """Load document with memory-efficient chunking."""
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        texts = self.text_splitter.split_documents(documents)
        return [doc.page_content for doc in texts]
    
    def process_document(self, file_path: str) -> list[dict[str, str]]:
        """Process document in batches to manage memory."""
        chunks = self.load_document(file_path)
        all_qa_pairs = []
        
        for chunk in tqdm(chunks, desc="Generating Q&A pairs"):
            qa_pairs = self.generate_qa_pairs(chunk)
            all_qa_pairs.extend(qa_pairs)
            
            # Clear memory after each chunk
            if hasattr(self.model, 'reset'):
                self.model.reset()
        
        return all_qa_pairs
    
    def save_qa_pairs(self, qa_pairs: list[dict[str, str]], output_path: str):
        """Save Q&A pairs with memory-efficient writing."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(f"{output_path}.json", 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(qa_pairs)
        df.to_csv(f"{output_path}.csv", index=False)
        
        # Save in JSONL format
        with open(f"{output_path}.jsonl", 'w') as f:
            for pair in qa_pairs:
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                }) + '\n')

# Example usage
if __name__ == "__main__":
    # First, check system and get recommendations
    optimizer = SystemOptimizer()
    settings = optimizer.get_optimal_settings()
    optimizer.print_recommendations()
    
    # Initialize generator with recommended settings
    generator = LaptopFriendlyQAGenerator(
        model_path=f"models/{settings['recommended_model']}",
        n_threads=settings['n_threads'],
        context_size=settings['context_size'],
        batch_size=settings['batch_size']
    )
    
    # Process document
    qa_pairs = generator.process_document("ndi-anomalies-advisories-aci.pdf")
    
    # Save results
    generator.save_qa_pairs(qa_pairs, "output/qa_pairs")

