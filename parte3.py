#Chatbot de Segmentação de Clientes Usando LLMs

# Imports
import pandas as pd
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Settings

# Configurando o título da página
st.set_page_config(page_title="Segmentação de Clientes", page_icon=":bar_chart:", layout="centered")

# Menu lateral
st.sidebar.title("Segmentação de Clientes")
st.sidebar.markdown("### Análise de Clientes com KMeans")
st.sidebar.button("AI Chatbot - Segmentação")

# Título da aplicação
st.title("Chatbot de Segmentação de Clientes Usando LLMs")

# Inicializa as mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com a análise do SuperStore Dataset?"}]

# Define o LLM
llm = Ollama(model="llama3.1")

# Modelo de embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Função para carregar os dados
@st.cache_resource()
def carregar_dados():
    with st.spinner("Carregando o arquivo df_final.csv..."):
        df = pd.read_csv("./final/df_final.csv")
        return df

# Função para criar o índice vetorial
@st.cache_resource()
def criar_indice(df):
    from llama_index.core import Document
    documentos = [Document(text=str(row)) for _, row in df.iterrows()]

    Settings.llm = llm
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(documentos)
    return index

# Carregar o dataframe e criar o índice
df = carregar_dados()
banco_vetorial = criar_indice(df)

# Inicializa o chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = banco_vetorial.as_chat_engine(chat_mode="condense_question", verbose=True)

# Captura a pergunta do usuário
if prompt := st.chat_input("Digite sua pergunta sobre os segmentos de clientes"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Exibe as mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Gera a resposta do assistente
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            user_message = st.session_state.messages[-1]["content"]
            contextual_prompt = ("Você é um analista de dados que recebeu a funçao de gerar insigths em linguagem natural e documentar a abordagem e resultados obtidos, nesse csv ja está feita a segmentação de clientes em 4 grupos, gere insigths relevantes e indicações: '" 
                                 f"{user_message}'. Responda de forma objetiva, explicando o segmento de clientes adequado.")
            response = st.session_state.chat_engine.chat(contextual_prompt)
            st.write(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

























































'''# Projeto 2 - Análise e Visualização de Dados de Vendas ao Longo do Tempo com PySpark
# Script de Treino do Modelo

# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml import Pipeline

# Inicializar a Spark Session
spark = SparkSession.builder \
    .appName("Projeto2-Treino") \
    .getOrCreate()

# Carregar os dados
data_path = '/opt/spark/dados/dataset.csv'
df_dsa = spark.read.csv(data_path, header=True, inferSchema=True)

# Ajustar o tipo de dado
df_dsa = df_dsa.withColumn('Date', col('Date').cast('date'))

# Extrair ano e mês da coluna 'Date'
df_dsa = df_dsa.withColumn('Year', year(col('Date'))).withColumn('Month', month(col('Date')))

# Assembler para transformar as colunas em vetor de features dentro do pipeline
feature_assembler = VectorAssembler(inputCols=['Year', 'Month'], outputCol='Features')

# Modelo de regressão linear
modelo_lr = LinearRegression(featuresCol='Features', labelCol='Sales')

# Configurar o pipeline com assembler e modelo
pipeline_dsa = Pipeline(stages=[feature_assembler, modelo_lr])

# Separar dados para treino e teste
dados_treino, dados_teste = df_dsa.randomSplit([0.7, 0.3])

# Treinar o modelo
modelo_dsa = pipeline_dsa.fit(dados_treino)

# Fazer previsões
previsoes = modelo_dsa.transform(dados_teste)
previsoes.select('Date', 'Sales', 'prediction').show()

# Avaliar o modelo
evaluator = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(previsoes)
print("\nRoot Mean Squared Error (RMSE) nos Dados de Teste = %g" % rmse)
print("\n")

evaluator = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(previsoes)
print("\nCoeficiente de Determinação (R2) nos Dados de Teste = %g" % r2)
print("\n")

# Salvar o modelo treinado
model_path = '/opt/spark/dados/modelo'
modelo_dsa.write().overwrite().save(model_path)

# Salvar as previsões em um arquivo CSV
previsoes.select('Date', 'Sales', 'prediction').write.csv('/opt/spark/dados/previsoesteste', header=True, mode="overwrite")

# Fechar a Spark session
spark.stop()'''

