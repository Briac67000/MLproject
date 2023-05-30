import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import numpy as np
from split_data import *
from preprocessing import *


# Préparer les données d'entraînement
df_input_unique_id_train=read_excel_df('C:\\Users\\moriceb\\Desktop\\MLproject\\input_train_group.xlsx')
df_summary_train=read_excel_df('C:\\Users\\moriceb\\Desktop\\MLproject\\summary_train_group.xlsx')

input_texts = list(df_input_unique_id_train)
summaries = list(df_summary_train)

# Charger le modèle pré-entraîné et le tokenizer
model_name = "Einmalumdiewelt/T5-Base_GNAD"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Définir les paramètres d'entraînement
num_epochs = 5
batch_size = 2
learning_rate = 1e-4
model.train()
# Boucle d'entraînement
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for i in range(0, len(input_texts), batch_size):
        batch_inputs = input_texts[i:i+batch_size]
        batch_summaries = summaries[i:i+batch_size]

        # Prétraiter les données
        inputs = tokenizer.batch_encode_plus(batch_inputs, return_tensors="pt",max_length=512, truncation=True, padding="longest")
        labels = tokenizer.batch_encode_plus(batch_summaries, return_tensors="pt", padding="longest")

        # Créer les tenseurs d'entrée et de sortie
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels_ids = labels["input_ids"]
        labels_attention_mask = labels["attention_mask"]

        # Passer les entrées à travers le modèle
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids, decoder_attention_mask=labels_attention_mask)

        # Calculer la perte
        loss = outputs.loss

        # Rétropropagation et mise à jour des poids
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# Évaluer le modèle (facultatif)

# Utiliser le modèle entraîné pour faire des prédictions
input_text = "Nouveau texte"
inputs = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=512, padding="longest")
encoder_inputs = inputs["input_ids"]
outputs = model.generate(input_ids=encoder_inputs, max_length=150, num_beams=4, early_stopping=True)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Summary:", translated_text)

if __name__ == "__main__":
    main()