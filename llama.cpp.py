from llama_cpp import Llama

llm = Llama(
      model_path="/home/youchengzhang/llm/qwen2.5-32b-instruct-q4_k_m/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      flash_attn=True,
      n_ctx=1024,
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
      "Q: 你是不是ai助手，用是或者否回答我 A: ", # Prompt
      max_tokens=1024, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      # echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output["choices"][0]["text"])
