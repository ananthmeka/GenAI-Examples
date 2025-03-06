import os
import psutil

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 ** 2  # Memory in MB
print(f"Interpreter Overhead Memory: {initial_memory:.2f} MB")

# Validation with Base Model 
#Load the Fine-Tuned Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

lib_memory = process.memory_info().rss / 1024 ** 2  # Memory in MB
print(f"Libraaries Memory: {lib_memory:.2f} MB")


from memory_profiler import profile

memory_after_memprofile = process.memory_info().rss / 1024 ** 2
print(f"Memory After memory_profiler Loaded: {memory_after_memprofile:.2f} MB")

# Example usage for a question and answer task
@profile
def answer_question(question, model, tokenizer):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@profile
def main():
    # Load the model and tokenizer from the saved directory
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # Example question
    question = "What is a correlated anomaly?"
    print(answer_question(question, model, tokenizer))

if __name__ == "__main__":
    main()

