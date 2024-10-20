# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from previous_chapters import ( #从先前的写的代码中，引入文本生成，模型构建函数。
    generate,
    GPTModel,
    text_to_token_ids,
    token_ids_to_text,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #检查是否有GPU，如果有则使用GPU，否则使用CPU。


def get_model_and_tokenizer(): #加载模型和分词器
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """

    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path(".") / "gpt2-medium355M-sft.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 7 code "
            " (ch07.ipynb) to generate the gpt2-medium355M-sft.pt file."
        )
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True) #载入模型权重
    model = GPTModel(GPT_CONFIG_355M) #创建模型实例
    model.load_state_dict(checkpoint) #加载模型权重
    model.to(device) #将模型放到GPU上

    return tokenizer, model, GPT_CONFIG_355M #返回分词器，模型，模型配置。


def extract_response(response_text, input_text): #提取模型生成的文本
    return response_text[len(input_text):].replace("### Response:", "").strip() #提取生成的文本，去掉前面的输入文本，去掉"### Response:"，去掉空格。


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer() #获取分词器，模型，模型配置。


@chainlit.on_message #接收用户输入的文本
async def main(message: chainlit.Message): #主函数（采用异步）
    """
    The main Chainlit function.
    """

    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content} 
    """

    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
