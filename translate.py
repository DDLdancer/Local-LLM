from llama_cpp import Llama

MAX_LENGTH=512
MAX_TOKENS=512
N_CTX=1024
PRINT=False

ENG_DELIMITERS=['.', '?', '!', '\n']
JP_DELIMITERS=['\n', '！', '？', '、', '。']
MODEL_PATH="/home/youchengzhang/llm/qwen2.5-32b-instruct-q4_k_m/model-qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf"
STOP="翻译完成"

# 初始化LLM模型
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    flash_attn=True,
    n_ctx=N_CTX,
)

def split_text_near_end_of_sentence(text, max_length=MAX_LENGTH, delimiters=JP_DELIMITERS):
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append((text, ''))
            break
        # 查找最近的句子结束符位置，并记录分隔符
        end_of_sentence = -1
        delimiter_used = ''
        for delim in delimiters:
            position = text.rfind(delim, 0, max_length)
            if position > end_of_sentence:
                end_of_sentence = position
                delimiter_used = delim
        
        if end_of_sentence == -1:  # 如果没有找到，使用最大长度
            end_of_sentence = max_length
            parts.append((text[:end_of_sentence], ''))
        else:
            end_of_sentence += 1  # 包括结束符本身
            parts.append((text[:end_of_sentence], delimiter_used))
        text = text[end_of_sentence:]
    return parts

def translate_text(text):
    # 将文本分割为适合模型的大小
    parts = split_text_near_end_of_sentence(text)
    translated_parts = []
    
    # 逐一翻译每部分
    for part, delim in parts:
        prompt = f"Q: 翻译如下文本到中文，翻译完成后输出“{STOP}”: {part} A: 以下是文本的中文翻译: "
        result = llm(prompt, max_tokens=MAX_TOKENS, stop=STOP)
        translated_text = result["choices"][0]["text"].strip()
        if delim == '\n':
            translated_text += '\n'  # 如果使用的分隔符是换行符，则在翻译文本后添加换行符
        if PRINT:
            print(part)
            print(translated_text)
        translated_parts.append(translated_text)
        
    # 返回拼接后的翻译文本
    return "".join(translated_parts)

# 读取文本文件
with open('input_article.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 翻译文本
translated_content = translate_text(content)

# 保存翻译结果
with open('translated_article.txt', 'w', encoding='utf-8') as file:
    file.write(translated_content)
