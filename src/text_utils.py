import numpy as np

def get_num_tokens(text, tokenizer):
    return len(tokenizer(text)['input_ids'])

def trim_text_to_token_limit(text, tokenizer,max_token_limit=4000):
    """
    Trims a large text by starting at a random paragraph and ensuring 
    the final text does not exceed the given token limit while ending at a complete paragraph.
    """
    # Split text into paragraphs (assuming paragraphs are separated by double newlines)
    paragraphs = text.strip().split("\n\n")
    
    # Shuffle to find a random valid starting paragraph
    # start_index = np.random.randint(0, len(paragraphs) )

    token_limit = np.random.uniform(500, 4000)
    start_index = 0
    
    # Initialize token count and final text
    final_text = []
    token_count = 0
    
    # Iterate through paragraphs starting from the selected random one
    for i in range(start_index, len(paragraphs)):
        paragraph = paragraphs[i]
        
        # Tokenize paragraph and count tokens
        paragraph_tokens = tokenizer(paragraph, return_tensors="pt", truncation=False)["input_ids"].shape[1]
        
        # Check if adding this paragraph exceeds the token limit
        if token_count + paragraph_tokens > token_limit:
            break
        
        # Append paragraph and update token count
        final_text.append(paragraph)
        token_count += paragraph_tokens
    
    return "\n\n".join(final_text)