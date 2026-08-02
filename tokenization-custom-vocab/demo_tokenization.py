import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    model_name = "distilbert/distilbert-base-uncased"
    print("=" * 60)
    print("PoC 1.3: Tokenization & Custom Vocabulary Extension Demo")
    print("=" * 60)

    # 1. Load Base Tokenizer & Model
    print(f"\n1. Loading base tokenizer and model: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    old_vocab_size = len(tokenizer)
    old_embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"   Original Tokenizer Vocab Size: {old_vocab_size}")
    print(f"   Original Model Embedding Shape: {model.get_input_embeddings().weight.shape}")

    # Sample domain-specific text with custom tags
    text = "[LOG_CRITICAL] [ENV_PROD] Connection pool exhausted on [API_GATEWAY]"
    print(f"\n   Target Text: \"{text}\"")

    # 2. Tokenize WITHOUT Custom Tokens
    tokens_before = tokenizer.tokenize(text)
    ids_before = tokenizer.encode(text)
    print("\n2. Tokenization BEFORE adding custom tokens:")
    print(f"   Tokens: {tokens_before}")
    print(f"   Token IDs: {ids_before}")
    print("   Notice how '[LOG_CRITICAL]' is fragmented into multiple subwords!")

    # 3. Add Custom Tokens
    custom_tokens = ["[LOG_CRITICAL]", "[ENV_PROD]", "[SYSTEM_ALERT]", "[API_GATEWAY]"]
    print(f"\n3. Adding {len(custom_tokens)} custom tokens: {custom_tokens}")
    num_added = tokenizer.add_tokens(custom_tokens)
    new_vocab_size = len(tokenizer)
    print(f"   Added {num_added} new tokens.")
    print(f"   New Tokenizer Vocab Size: {new_vocab_size}")

    # 4. Tokenize AFTER Adding Custom Tokens
    tokens_after = tokenizer.tokenize(text)
    ids_after = tokenizer.encode(text)
    print("\n4. Tokenization AFTER adding custom tokens:")
    print(f"   Tokens: {tokens_after}")
    print(f"   Token IDs: {ids_after}")
    print("   Each custom tag is now treated as a single, discrete token!")

    # 5. Resize Model Embeddings
    print("\n5. Resizing Model Embedding Matrix...")
    print(f"   Shape before resize: {model.get_input_embeddings().weight.shape}")
    model.resize_token_embeddings(len(tokenizer))
    new_embedding_shape = model.get_input_embeddings().weight.shape
    print(f"   Shape after resize:  {new_embedding_shape}")

    # 6. Apply Mean Embedding Initialization to New Token Vectors
    print("\n6. Initializing New Token Embedding Weights...")
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        old_embeddings_matrix = embeddings.weight[:old_vocab_size]
        mean_embedding_vec = old_embeddings_matrix.mean(dim=0)

        for token_id in range(old_vocab_size, new_vocab_size):
            embeddings.weight[token_id] = mean_embedding_vec.clone()

        print(f"   Successfully initialized new token embeddings (IDs {old_vocab_size} to {new_vocab_size-1})")
        print(f"   Mean vector sample (first 5 dims): {mean_embedding_vec[:5].tolist()}")
        print(f"   New token ID {old_vocab_size} ('{tokenizer.decode([old_vocab_size])}') embedding sample (first 5 dims): {embeddings.weight[old_vocab_size][:5].tolist()}")

    print("\n" + "=" * 60)
    print("Demo complete! All vocabulary and embedding operations executed smoothly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
