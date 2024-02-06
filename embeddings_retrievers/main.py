from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# Experiements trying to run local model with GPU support
# /Program Files/CodeBlocks/MinGW/bin/gcc.exe
# /Program Files/CodeBlocks/MinGW/bin/g++.exe

# $env:CMAKE_ARGS = "-DLLAMA_HIPBLAS=1 -DLLAMA_CLBLAST=1 -DLLAMA_OPENBLAS=1 -j6 -DCMAKE_C_COMPILER=C:/Development/CodeBlocks/MinGW/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/Development/CodeBlocks/MinGW/bin/g++.exe"

# E_CXX_COMPILER=C:/Development/CodeBlocks/MinGW/bin/g++.exe"
# pip install llama-cpp-python  --upgrade --force-reinstall --no-cache-dir

# llm = LlamaCpp(
#     model_path="/Users/Anthony/Downloads/LLM Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     n_gpu_layers=20,
#     n_batch=512,
#     n_ctx=2048,
#     f16_kv=True,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=True,
# )

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code: \n{code}",
    input_variables=["language", "code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

result = chain({
    "language": args.language,
    "task": args.task,
})

print(">>>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>>> GENERATED TEST:")
print(result["test"])
