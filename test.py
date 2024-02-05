from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load the pre-trained model and tokenizer
def get_tokenizer_model(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/',
                                                  use_auth_token=auth_token, trust_remote_code=True, device_map="cuda") 

    return tokenizer, model


if '__main__' == __name__:

    print(torch.cuda.is_available())

    name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer, model = get_tokenizer_model(name)

    print(model)

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300,
    )

    prompt_template = """[INST] {question} [/INST] """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    llm = HuggingFacePipeline(pipeline=pipeline) 
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)

    response = chain.run("say : it works, in French")

    print(response)