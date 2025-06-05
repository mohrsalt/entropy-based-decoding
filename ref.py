from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from tqdm import tqdm

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=quantization_config,
        torch_dtype="auto"
    )
prompts = [
    "I would like to",
    "hello how are you",
    "what is going on",
    "roses are red and",
    "welcome to the hotel",
]

distributed_state = PartialState()
#model.to(distributed_state.device)

batch_size = 2
pad_to_multiple_of = 8 
tokenizer.pad_token = tokenizer.eos_token

# split into batch
formatted_prompts = [
    prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
]

padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"

# tokenize each batch
tokenized_prompts = [
    tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
    for formatted_prompt in formatted_prompts
]

completions_per_process = []
with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
    for batch in tqdm(batched_prompts, desc=f"Generating completions on device {distributed_state.device}"):
        # move the batch to the correct 
        batch = batch.to(distributed_state.device)
        outputs = model.generate(**batch, max_new_tokens=20)
        outputs = [output[len(prompt) :] for prompt, output in zip(batch["input_ids"], outputs)]
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        completions_per_process.extend(generated_text)
        
completions_gather = gather_object(completions_per_process)
# Drop duplicates produced by apply_padding in  split_between_processes
completions = completions_gather[: len(prompts)]
# Reset tokenizer padding side
tokenizer.padding_side = padding_side_default
if distributed_state.is_main_process:
    print(completions)
