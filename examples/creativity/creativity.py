from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt

from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()

# Initialize SmolLM2 model and tokenizer (135M parameters)
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)





creative_examples = [
    "The clock melted into Tuesday, dripping minutes onto the sidewalk where dreams collect in puddles."
    "She painted with sounds, each brushstroke a whispered symphony that only the walls could hear."
    "The library's books began to migrate south for winter, their pages fluttering like paper wings."
    "His thoughts were origami cranes folding themselves into tomorrow's possibilities."
    "The moon borrowed a ladder from the stars to peek into bedroom windows and collect forgotten wishes."
]

non_creative_examples = [
    "The meeting is scheduled for 3 PM in conference room B. Please bring your quarterly reports."
    "To make coffee, add hot water to ground coffee beans and let it steep for four minutes."
    "The weather forecast shows rain tomorrow with temperatures reaching 65 degrees Fahrenheit."
    "Please submit your expense reports by Friday. Include all receipts and proper documentation."
    "The store is open Monday through Saturday from 9 AM to 6 PM. We accept cash and credit cards."
]

template_str = '{user_tag} Judge the creativity of the following piece of writing:\nPiece of writing: {example}\nAnswer: {assistant_tag} '

USER_TAG = '[USER]'
ASSISTANT_TAG = '[ASSISTANT]'

data = [[p,n] for p,n in zip(creative_examples, non_creative_examples)]
labels = []
for d in data:
    true_s = d[0]
    random.shuffle(d)
    labels.append([s == true_s for s in d])

data = np.concatenate(data).tolist()

print(data[:2])

formatted_data = [template_str.format( example=d, user_tag=USER_TAG, assistant_tag=ASSISTANT_TAG) for d in data] 


split_idx = 4

train_data = formatted_data[:split_idx]
train_labels = labels[:split_idx]

test_data = formatted_data[split_idx:]
test_labels = labels[split_idx:]






# Todo apply template

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'

direction_finder_kwargs={"n_components": 1}

# Example usage of the pipelines
creativity_rep_reader = rep_reading_pipeline.get_directions(
    train_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=n_difference,
    train_labels=train_labels,
    direction_method=direction_method,
    direction_finder_kwargs=direction_finder_kwargs
)

# Get hidden states for test data
H_tests = rep_reading_pipeline(
        test_data, 
        rep_token=rep_token, 
        hidden_layers=hidden_layers, 
        rep_reader=creativity_rep_reader,
        batch_size=32)
    

results = {layer: {} for layer in hidden_layers}
rep_readers_means = {}
rep_readers_means['creativity'] = {layer: 0 for layer in hidden_layers}

for layer in hidden_layers:
    H_test = [H[layer] for H in H_tests]
    rep_readers_means['creativity'][layer] = np.mean(H_test)
    # Group into pairs (creative, non-creative)
    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]

    sign = creativity_rep_reader.direction_signs[layer]

    eval_func = min if sign == -1 else max
    # For each pair, check if the creative example (H[0]) has the expected higher/lower value
    cors = np.mean([eval_func(H) == H[0] for H in H_test])
    results[layer] = cors

print(results)
plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
plt.show()




# Set up control kwargs (placeholder - adjust based on your needs)
# control_kwargs = {}
#rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)
