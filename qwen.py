from transformers import AutoModelForCausalLM, AutoTokenizer
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)  # 初始化colorama，并设置样式自动重置

# 模型名称
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化对话
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
]

# 交互式对话循环
try:
    while True:
        print(Fore.CYAN + "Enter your prompt (Press Ctrl+D or Ctrl+Z followed by Enter to finish the input):")
        input_text = []
        try:
            while True:
                line = input()
                input_text.append(line)
        except EOFError:
            pass
        prompt = '\n'.join(input_text)
        
        # 更新消息列表
        messages.append({"role": "user", "content": prompt})

        # 应用对话模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print(Fore.YELLOW + "Generating response...")
        # 生成回应
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 输出回应
        print(Fore.GREEN + "Qwen:" + Style.RESET_ALL, response)

        # 将回应添加到消息列表
        messages.append({"role": "assistant", "content": response})
except KeyboardInterrupt:
    print(Fore.RED + "Exiting the chat.")

