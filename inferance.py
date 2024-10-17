from model import TransformerLightningModel, TransformerModel
import torch
from transformers import GPT2Tokenizer
import pytorch_lightning as pl

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

model = TransformerModel(
    vocab_size=tokenizer.vocab_size, 
    model_dim=512, 
    num_heads=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6, 
    dim_feedforward=2048, 
    dropout=0.1
)

checkpoint_path = "final_model.ckpt"
trained_model = TransformerLightningModel.load_from_checkpoint(checkpoint_path, model=model)

# Ensure model is in evaluation mode for inference
trained_model.eval()

def generate_story(model, tokenizer, prompt, max_length=100):
    # Encode the input prompt into input IDs
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    
    # Move the input_ids to the same device as the model
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Autoregressively generate text
    for _ in range(max_length):
        # Generate the next token logits
        with torch.no_grad():
            # Forward pass through the model
            outputs = model(input_ids, input_ids)  # Passing input through both encoder and decoder
        
        # Since outputs is 2D, it has shape (batch_size, vocab_size)
        next_token_logits = outputs[-1]  # Get logits for the last token directly
        
        # Get the most likely next token
        next_token = torch.argmax(next_token_logits, dim=-1)

        # Ensure next_token has shape (batch_size, 1)
        next_token = next_token.unsqueeze(1)  # Add sequence dimension

        # Concatenate next_token to the input_ids along the sequence dimension (dim=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Generate a story from a prompt
prompt = "Tell a story"
generated_story = generate_story(trained_model, tokenizer, prompt)
print("Generated story:\n", generated_story)