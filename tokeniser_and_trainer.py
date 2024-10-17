from datasets import load_dataset
from transformers import GPT2Tokenizer
import pytorch_lightning as pl
from model import TransformerModel, TransformerLightningModel
from datasets import load_dataset
from transformers import GPT2Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Tokenize dataset with padding and truncation
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        return_tensors='pt',
        truncation=True,
        padding='max_length',  # or 'longest'
        max_length=512  # Adjust max_length according to your needs
    )

# Map tokenization function over the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids'])

# Create DataLoader
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=8, shuffle=True)

model = TransformerModel(
    vocab_size=tokenizer.vocab_size, 
    model_dim=512, 
    num_heads=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6, 
    dim_feedforward=2048, 
    dropout=0.1
)

lightning_model = TransformerLightningModel(model)

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',  # Metric to monitor
    dirpath='checkpoints/',  # Directory to save checkpoints
    filename='best-checkpoint',  # Name of the checkpoint file
    save_top_k=1,  # Save only the best model
    mode='min',  # Save the model with the minimum validation loss
)

# Create a trainer with the checkpoint callback
trainer = Trainer(
    max_epochs=3,
    callbacks=[checkpoint_callback]
)

# Train the model
trainer.fit(lightning_model, train_dataloader)

trainer.save_checkpoint("final_model.ckpt")

##INFERANCE

def generate(model, tokenizer, prompt, max_len=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    output = model(input_ids, input_ids)  # Autoregressive generation can be refined
    return tokenizer.decode(output[0].argmax(dim=-1), skip_special_tokens=True)

prompt = "Write a short story about something you have been trained on"
generated_text = generate(model, tokenizer, prompt)
print(generated_text)