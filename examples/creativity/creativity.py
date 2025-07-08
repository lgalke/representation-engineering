from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt

from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()

# Initialize SmolLM2 model and tokenizer (135M parameters)
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token_id = model.config.eos_token_id

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)


dummy_data = [
    # Creative examples
    {"text": "The clock melted into Tuesday, dripping minutes onto the sidewalk where dreams collect in puddles.", "label": 1},
    {"text": "She painted with sounds, each brushstroke a whispered symphony that only the walls could hear.", "label": 1},
    {"text": "The library's books began to migrate south for winter, their pages fluttering like paper wings.", "label": 1},
    {"text": "His thoughts were origami cranes folding themselves into tomorrow's possibilities.", "label": 1},
    {"text": "The moon borrowed a ladder from the stars to peek into bedroom windows and collect forgotten wishes.", "label": 1},
    # Non-creative examples
    {"text": "The meeting is scheduled for 3 PM in conference room B. Please bring your quarterly reports.", "label": 0},
    {"text": "To make coffee, add hot water to ground coffee beans and let it steep for four minutes.", "label": 0},
    {"text": "The weather forecast shows rain tomorrow with temperatures reaching 65 degrees Fahrenheit.", "label": 0},
    {"text": "Please submit your expense reports by Friday. Include all receipts and proper documentation.", "label": 0},
    {"text": "The store is open Monday through Saturday from 9 AM to 6 PM. We accept cash and credit cards.", "label": 0}
]






# Convert to HuggingFace Dataset
dataset = Dataset.from_list(dummy_data)

positive = dataset[:5]
negative = dataset[5:]

# TODO data setup



rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'


# Example usage of the pipelines

print(dataset[:2])
creativity_rep_reader = rep_reading_pipeline.get_directions(
    dataset['text'],
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=n_difference,
    train_labels=dataset['label'],
    direction_method=direction_method,
    batch_size=32,
)


# Get hidden states for test data
H_tests = creativity_rep_reader.get_rep_acts(dataset['contrast_example'], rep_token=rep_token, hidden_layers=hidden_layers)

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

plt.plot(hidden_layers, [results[layer] for layer in hidden_layers])
plt.show()





# Set up control kwargs (placeholder - adjust based on your needs)
# control_kwargs = {}
#rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)
