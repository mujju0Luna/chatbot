from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name):
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"./{model_name}_tokenizer")

    # Download the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(f"./{model_name}_model")

    print(f"Model and tokenizer for '{model_name}' downloaded and saved locally.")

if __name__ == "__main__":
    # Specify the EleutherAI GPT-J model
    model_name = "EleutherAI/gpt-j-6B"
    download_model(model_name)
