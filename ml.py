!pip install torch pandas
!pip install transformers accelerate
!pip install tokenizers




import os
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import GPT2TokenizerFast as GPT2Tokenizer
from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Specify the folder in Google Drive
folder = '/content/drive/My Drive'

# Specify the name of the model, you can change this to `gpt2-medium`, `gpt2-large`, etc.
model_name = 'gpt2'

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the padding token is setup correctly
tokenizer.pad_token = tokenizer.eos_token

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(inputs['input_ids'], dtype=torch.long)
        }


# Load the dataset
df = pd.read_csv(os.path.join(folder, 'messages.csv'))
texts = df['message_text'].tolist()
max_len = 512
train_dataset = TextDataset(texts, tokenizer, max_len)

# Define the training arguments
# Define the training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(folder, 'results'),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    push_to_hub=False,
    load_best_model_at_end=False,  # Set this to False
    save_strategy="steps",
    evaluation_strategy="no"
)




# Create the trainer
# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]), 
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]), 
                                'labels': torch.stack([f['labels'] for f in data])},
)


# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.join(folder, 'model'))

# Save the tokenizer
tokenizer.save_pretrained(os.path.join(folder, 'model'))
