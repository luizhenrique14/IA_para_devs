import os
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
from datasets import Dataset

# -----------------------------
# 1. CONFIGURAÇÕES E PREPARAÇÃO
# -----------------------------

MODEL_NAME = "microsoft/phi-2"  # modelo mais robusto com melhor compreensão
TRAIN_FILE = "dados_treinamento_new.txt"
QUESTIONS_FILE = "perguntas.txt"

def baixar_modelo():
    print("🔽 Baixando modelo pré-treinado...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Configurando o padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return tokenizer, model

# -----------------------------
# 2. FUNÇÃO PARA CONSULTAR MODELO
# -----------------------------

def criar_chain(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.01,  # Temperatura extremamente baixa para respostas mais consistentes
        top_p=0.95,
        no_repeat_ngram_size=3,
        do_sample=False,  # Desabilita amostragem aleatória
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2  # Penaliza repetições
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    template = """Instruções: Você é um especialista em mercado financeiro brasileiro. Sua tarefa é fornecer informações precisas e detalhadas sobre o sistema financeiro do Brasil.

Pergunta: {pergunta}

Responda de forma direta e objetiva, usando apenas informações factuais baseadas no seu conhecimento sobre o mercado financeiro brasileiro. Evite opiniões pessoais ou especulações.

Resposta: """
    prompt = PromptTemplate(template=template, input_variables=["pergunta"])
    return LLMChain(prompt=prompt, llm=llm)

# -----------------------------
# 3. EXECUTAR PERGUNTAS
# -----------------------------

def fazer_perguntas(chain, perguntas):
    respostas = []
    for p in perguntas:
        print(f"\n❓ Pergunta: {p.strip()}")
        resposta = chain.run(pergunta=p)
        respostas.append((p, resposta.strip()))
        print(f"💬 Resposta: {resposta.strip()}")
    return respostas

# -----------------------------
# 4. FINE-TUNING DO MODELO
# -----------------------------

def treinar_modelo(model, tokenizer, dados):
    print("\n🧩 Iniciando fine-tuning com dados fictícios...")

    # Criando dataset simples (cada linha = exemplo)
    dataset = Dataset.from_dict({"text": dados})

    def tokenize_function(examples):
        # Tokenização com labels para language modeling
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # Aumentando o tamanho máximo para capturar mais contexto
            return_tensors="pt",
            return_special_tokens_mask=True
        )
        result["labels"] = result["input_ids"].clone()
        # Mascara tokens especiais para não serem usados no cálculo da loss
        for i, mask in enumerate(result["special_tokens_mask"]):
            for j, val in enumerate(mask):
                if val == 1:
                    result["labels"][i][j] = -100
        return result

    # Pré-processar os dados
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_datasets.set_format("torch")

    # Configuração do treinamento
    training_args = TrainingArguments(
        output_dir="./resultados",
        per_device_train_batch_size=1,
        num_train_epochs=30,  # Mais épocas para melhor aprendizado
        save_steps=5,
        logging_steps=5,
        save_total_limit=2,
        learning_rate=5e-6,  # Taxa de aprendizado menor para treinamento mais estável
        weight_decay=0.01,
        warmup_ratio=0.1,  # Usa ratio em vez de steps fixos
        logging_dir="./logs",
        fp16=False,
        gradient_accumulation_steps=32,  # Aumentado para melhor estabilidade
        remove_unused_columns=False,
        learning_rate_schedule="linear",
        max_grad_norm=0.5  # Limita o gradiente para evitar explosão
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()
    print("✅ Fine-tuning concluído!")
    return model

# -----------------------------
# 5. EXECUÇÃO PRINCIPAL
# -----------------------------

if __name__ == "__main__":
    # Garante que os arquivos de teste existam
    if not os.path.exists(TRAIN_FILE):
        with open(TRAIN_FILE, "w", encoding="utf-8") as f:
            f.write("""A taxa Selic é a taxa básica de juros da economia brasileira.
A inflação é medida pelo IPCA e reflete o aumento geral dos preços.
O Banco Central é responsável pela política monetária no Brasil.
O mercado financeiro é influenciado pela taxa de câmbio, juros e inflação.""")
    if not os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
            f.write("""O que é a taxa Selic?
Quem define a política monetária no Brasil?
O que influencia o mercado financeiro brasileiro?""")

    # Carregar perguntas
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        perguntas = f.readlines()

    # 1️⃣ Baixar modelo
    tokenizer, model = baixar_modelo()

    # 2️⃣ Chain inicial (antes do treino)
    print("\n=== PERGUNTAS ANTES DO FINE-TUNING ===")
    chain_inicial = criar_chain(model, tokenizer)
    respostas_antes = fazer_perguntas(chain_inicial, perguntas)

    # 3️⃣ Ler dados e treinar
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        dados_treinamento = f.readlines()

    model_treinado = treinar_modelo(model, tokenizer, dados_treinamento)

    # 4️⃣ Chain após o fine-tuning
    print("\n=== PERGUNTAS APÓS O FINE-TUNING ===")
    chain_final = criar_chain(model_treinado, tokenizer)
    respostas_depois = fazer_perguntas(chain_final, perguntas)

    # 5️⃣ Comparar resultados
    print("\n📊 COMPARATIVO:")
    for i, (p, _) in enumerate(respostas_antes):
        print(f"\nPergunta: {perguntas[i].strip()}")
        print(f"Antes: {respostas_antes[i][1]}")
        print(f"Depois: {respostas_depois[i][1]}")
