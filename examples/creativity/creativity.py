from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()

# Initialize SmolLM2 model and tokenizer (135M parameters)
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up control kwargs (placeholder - adjust based on your needs)
control_kwargs = {}

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)

dummy_data = [
    # Creative examples
    {"text": "The clock melted into Tuesday, dripping minutes onto the sidewalk where dreams collect in puddles.", "label": "creative"},
    {"text": "She painted with sounds, each brushstroke a whispered symphony that only the walls could hear.", "label": "creative"},
    {"text": "The library's books began to migrate south for winter, their pages fluttering like paper wings.", "label": "creative"},
    {"text": "His thoughts were origami cranes folding themselves into tomorrow's possibilities.", "label": "creative"},
    {"text": "The moon borrowed a ladder from the stars to peek into bedroom windows and collect forgotten wishes.", "label": "creative"},
    
    # Non-creative examples
    {"text": "The meeting is scheduled for 3 PM in conference room B. Please bring your quarterly reports.", "label": "non-creative"},
    {"text": "To make coffee, add hot water to ground coffee beans and let it steep for four minutes.", "label": "non-creative"},
    {"text": "The weather forecast shows rain tomorrow with temperatures reaching 65 degrees Fahrenheit.", "label": "non-creative"},
    {"text": "Please submit your expense reports by Friday. Include all receipts and proper documentation.", "label": "non-creative"},
    {"text": "The store is open Monday through Saturday from 9 AM to 6 PM. We accept cash and credit cards.", "label": "non-creative"}
]


# Example usage of the pipelines
rep_reading_results = rep_reading_pipeline(dummy_data)
print(rep_reading_results)







