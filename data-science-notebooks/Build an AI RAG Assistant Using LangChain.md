

**Build an AI RAG Assistant Using LangChain**



\#### The project integrates multiple AI techniques to build a robust document querying system capable of processing various types of documents and answering questions based on their content. It involves several steps, including document loading, text chunking, embedding generation using models like sentence-transformers/all-MiniLM-L6-v2, and a similarity search. Key components include utilizing LangChain for document loading and text processing, Chroma for vector databases, and HuggingFace models for text generation. Gradio is used for creating an interactive interface that allows users to upload documents and ask questions, while the system retrieves relevant information and generates insightful responses. The project showcases the combination of RAG (Retrieval-Augmented Generation) techniques for enhancing document-based AI applications.



**Table of Contents:**

\
\
1\. Installing Required Libraries.\
2\. Task 1: Load Document Using LangChain for Different Sources.\
3\. Task 2: Apply Text Splitting Techniques.\
4\. Task 3: Embed Documents.\
5\. Task 4: Create and Configure Vector Databases to Store Embeddings.\
6\. Task 5: Develop a Retriever to Fetch Document Segments Based on Queries.\
7\. Task 6: Construct a QA Bot That Leverages LangChain and LLM to Answer Questions.

**In [1]:**

%%capture

!pip install langchain

!pip install langchain-community

!pip install chromadb

!pip install sentence-transformers

!pip install transformers

!pip install torch

!pip install pypdf

!pip install gradio

**In [2]:**

def warn(\*args, \*\*kwargs):

`    `pass

import warnings

warnings.warn = warn

warnings.filterwarnings('ignore')



**Task 1: Load document using LangChain for different sources**

**In [4]:**

from langchain\_community.document\_loaders import PyPDFLoader, TextLoader

\# Specify the path to the PDF file

pdf\_file = "A\_Comprehensive\_Review\_of\_Low\_Rank\_Adaptation\_in\_Large\_Language\_Models\_for\_Efficient\_Parameter\_Tuning-1.pdf"

\# Create an instance of PyPDFLoader to load the PDF file

loader = PyPDFLoader(pdf\_file)

\# Load the content of the PDF file into a list of documents (each page is treated as a separate document)

docs = loader.load()

\# Print the first 1000 characters of the content from the first page of the PDF

print(docs[0].page\_content[:1000])

A Comprehensive Review of Low-Rank

Adaptation in Large Language Models for

Efficient Parameter Tuning

September 10, 2024

Abstract

Natural Language Processing (NLP) often involves pre-training large

models on extensive datasets and then adapting them for specific tasks

through fine-tuning. However, as these models grow larger, like GPT-3

with 175 billion parameters, fully fine-tuning them becomes computa-

tionally expensive. We propose a novel method called LoRA (Low-Rank

Adaptation) that significantly reduces the overhead by freezing the orig-

inal model weights and only training small rank decomposition matrices.

This leads to up to 10,000 times fewer trainable parameters and reduces

GPU memory usage by three times. LoRA not only maintains but some-

times surpasses fine-tuning performance on models like RoBERTa, De-

BERTa, GPT-2, and GPT-3. Unlike other methods, LoRA introduces

no extra latency during inference, making it more efficient for practical

applications. All relevant code an



**Task 2: Apply text splitting techniques**

**In [5]:**

latex\_text = r"""

`    `\documentclass{article}

`    `\begin{document}

`    `\maketitle

`    `\section{Introduction}

`    `Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.

`    `\subsection{History of LLMs}

The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

\subsection{Applications of LLMs}

LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

\end{document}

"""

**In [6]:**

from langchain.text\_splitter import Language, RecursiveCharacterTextSplitter

\# Initializing the RecursiveCharacterTextSplitter for LaTeX documents.

\# This splitter breaks the text into chunks based on LaTeX syntax rules.

latex\_splitter = RecursiveCharacterTextSplitter.from\_language(

`    `language=Language.LATEX,  # Specifying the language as LaTeX for proper text segmentation

`    `chunk\_size=60,            # Defining the maximum number of characters in each chunk

`    `chunk\_overlap=0           # Setting overlap between chunks to zero to avoid duplication

)

\# Splitting the LaTeX text into smaller chunks (documents).

\# The splitter processes the LaTeX structure, maintaining logical separation.

latex\_docs = latex\_splitter.create\_documents([latex\_text])

\# Looping through the generated chunks to display their content.

\# This helps verify how the LaTeX code has been split.

for doc in latex\_docs:

`    `print(doc.page\_content)

\documentclass{article}

`    `\begin{document}

\maketitle

`    `\section{Introduction}

`    `Large language

models (LLMs) are a type of machine learning model that can

be trained on vast amounts of text data to generate

human-like language. In recent years, LLMs have made

significant advances in various natural language processing

tasks, including language translation, text generation, and

sentiment analysis.

`    `\subsection{History of LLMs}

The

earliest LLMs were developed in the 1980s and 1990s, but

they were limited by the amount of data that could be

processed and the computational power available at the

time. In the past decade, however, advances in hardware and

software have made it possible to train LLMs on massive

datasets, leading to significant improvements in

performance.

\subsection{Applications of LLMs}

LLMs have many

applications in the industry, including chatbots, content

creation, and virtual assistants. They can also be used in

academia for research in linguistics, psychology, and

computational linguistics.

\end{document}



**Task 3: Embed documents**

**In [7]:**

from sentence\_transformers import SentenceTransformer

\# Initializing the SentenceTransformer model with a pre-trained model ("all-MiniLM-L6-v2") for embedding text

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

\# Defining a query (text) that will be encoded into an embedding representation

query = "How are you?"

\# Encoding the query into an embedding using the model

embedding = model.encode(query)

\# Printing the first 5 values of the embedding vector to check the result

print(embedding[:5])

modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]

config\_sentence\_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]

README.md:   0%|          | 0.00/10.7k [00:00<?, ?B/s]

sentence\_bert\_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]

config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]

model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]

tokenizer\_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]

vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]

tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]

special\_tokens\_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]

1\_Pooling/config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]

[0.00700396 0.01091414 0.08746253 0.08679935 0.02664845]



**Task 4: Create and configure vector databases to store embeddings**

**In [8]:**

from langchain\_community.document\_loaders import TextLoader

from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings

\# Loading the text document from the file "new\_Policies.txt"

loader = TextLoader("new\_Policies.txt")

documents = loader.load()

\# Initializing the HuggingFaceEmbeddings model with a pre-trained model for embedding (MiniLM)

embedding\_function = HuggingFaceEmbeddings(model\_name="sentence-transformers/all-MiniLM-L6-v2")

\# Creating a Chroma vector store with documents and embedding function

vectordb = Chroma.from\_documents(documents, embedding\_function)

\# Defining a query ("Smoking policy") to search within the documents

query = "Smoking policy"

\# Performing a similarity search for the query, retrieving the top 1 document

docs = vectordb.similarity\_search(query, k=1)

\# Printing the search result

for i, doc in enumerate(docs):

`    `print(f"Result {i+1}:\n{doc.page\_content}\n{'-'\*50}")

Result 1:

1\. Code of Conduct

Our Code of Conduct establishes the core values and ethical standards that all members of our organization must adhere to. We are committed to fostering a workplace characterized by integrity, respect, and accountability.

Integrity: We commit to the highest ethical standards by being honest and transparent in all our dealings, whether with colleagues, clients, or the community. We protect sensitive information and avoid conflicts of interest.

Respect: We value diversity and every individual's contribution. Discrimination, harassment, or any form of disrespect is not tolerated. We promote an inclusive environment where differences are respected, and everyone is treated with dignity.

Accountability: We are responsible for our actions and decisions, complying with all relevant laws and regulations. We aim for continuous improvement and report any breaches of this code, supporting investigations into such matters.

Safety: We prioritize the safety of our employees, clients, and the community. We encourage a culture of safety, including reporting any unsafe practices or conditions.

Environmental Responsibility: We strive to reduce our environmental impact and promote sustainable practices.

This Code of Conduct is the cornerstone of our organizational culture. We expect every employee to uphold these principles and act as role models, ensuring our reputation for ethical conduct, integrity, and social responsibility.

2\. Recruitment Policy

Our Recruitment Policy is dedicated to attracting, selecting, and integrating the most qualified and diverse candidates into our organization. The success of our company depends on the talent, skills, and commitment of our employees.

Equal Opportunity: We are an equal opportunity employer and do not discriminate based on race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or any other protected status. We actively support diversity and inclusion.

Transparency: We maintain a transparent recruitment process. Job vacancies are advertised both internally and externally when appropriate. Job descriptions and requirements are clear and accurately reflect the role.

Selection Criteria: We base our selection on qualifications, experience, and skills relevant to the role. Our interviews and assessments are objective, and decisions are made impartially.

Data Privacy: We are dedicated to protecting candidates' personal information and comply with all applicable data protection laws.

Feedback: Candidates receive timely and constructive feedback on their applications and interview performance.

Onboarding: New hires receive thorough onboarding to help them integrate effectively, including an overview of our culture, policies, and expectations.

Employee Referrals: We welcome employee referrals as they help build a strong and engaged team.

This policy lays the foundation for a diverse, inclusive, and talented workforce. It ensures that we hire candidates who align with our values and contribute to our success. We regularly review and update this policy to incorporate best practices in recruitment.


3\. Internet and Email Policy

Our Internet and Email Policy ensures the responsible and secure use of these tools within our organization, recognizing their importance in daily operations and the need for compliance with security, productivity, and legal standards.

Acceptable Use: Company-provided internet and email are primarily for job-related tasks. Limited personal use is permitted during non-work hours as long as it does not interfere with work duties.

Security: Protect your login credentials and avoid sharing passwords. Be cautious with email attachments and links from unknown sources, and promptly report any unusual online activity or potential security threats.

Confidentiality: Use email for confidential information, trade secrets, and sensitive customer data only with encryption. Be careful when discussing company matters on public platforms or social media.

Harassment and Inappropriate Content: Internet and email must not be used for harassment, discrimination, or the distribution of offensive content. Always communicate respectfully and sensitively online.

Compliance: Adhere to all relevant laws and regulations concerning internet and email use, including copyright and data protection laws.

Monitoring: The company reserves the right to monitor internet and email usage for security and compliance purposes.

Consequences: Violations of this policy may lead to disciplinary action, including potential termination.

This policy promotes the safe and responsible use of digital communication tools in line with our values and legal obligations. Employees must understand and comply with this policy. Regular reviews will ensure it remains relevant with changing technology and security standards.

4\. Mobile Phone Policy

Our Mobile Phone Policy defines standards for responsible use of mobile devices within the organization to ensure alignment with company values and legal requirements.

Acceptable Use: Mobile devices are primarily for work-related tasks. Limited personal use is allowed if it does not disrupt work responsibilities.

Security: Secure your mobile device and credentials. Be cautious with app downloads and links from unknown sources, and report any security issues promptly.

Confidentiality: Avoid sharing sensitive company information via unsecured messaging apps or emails. Exercise caution when discussing company matters in public.

Cost Management: Personal use of mobile phones should be separate from company accounts, and any personal charges on company-issued phones must be reimbursed.

Compliance: Comply with all relevant laws and regulations concerning mobile phone usage, including data protection and privacy laws.

Lost or Stolen Devices: Immediately report any lost or stolen mobile devices to the IT department or your supervisor.

Consequences: Non-compliance with this policy may result in disciplinary actions, including potential loss of mobile phone privileges.

This policy encourages the responsible use of mobile devices in line with legal and ethical standards. Employees are expected to understand and follow these guidelines. The policy is regularly reviewed to stay current with evolving technology and security best practices.

\--------------------------------------------------



**Task 5: Develop a retriever to fetch document segments based on queries**

**In [9]:**

\# Loading the text document "new\_Policies.txt"

loader = TextLoader("new\_Policies.txt")

documents = loader.load()

\# Initializing a text splitter that splits documents into chunks of 500 characters with 100 characters of overlap

text\_splitter = RecursiveCharacterTextSplitter(chunk\_size=500, chunk\_overlap=100)

\# Splitting the documents into chunks

chunks\_txt = text\_splitter.split\_documents(documents)

\# Initializing the HuggingFaceEmbeddings model to create embeddings using the MiniLM model

embedding\_function = HuggingFaceEmbeddings(model\_name="sentence-transformers/all-MiniLM-L6-v2")

\# Creating a Chroma vector store from the chunks of text and the embedding function

vectordb = Chroma.from\_documents(documents=chunks\_txt, embedding=embedding\_function)

\# Setting up a retriever to retrieve the top 2 most similar documents for a given query

retriever = vectordb.as\_retriever(search\_kwargs={"k": 2})

\# Defining the search query ("Email policy")

query = "Email policy"

\# Retrieving the top 2 documents based on the similarity to the query

docs = retriever.invoke(query)

\# Printing the content of the retrieved documents (up to the first 500 characters)

for i, doc in enumerate(docs):

`    `print(f"Result {i+1}:\n{doc.page\_content[:500]}...\n")

Result 1:

3\. Internet and Email Policy

Our Internet and Email Policy ensures the responsible and secure use of these tools within our organization, recognizing their importance in daily operations and the need for compliance with security, productivity, and legal standards.

Acceptable Use: Company-provided internet and email are primarily for job-related tasks. Limited personal use is permitted during non-work hours as long as it does not interfere with work duties....

Result 2:

Harassment and Inappropriate Content: Internet and email must not be used for harassment, discrimination, or the distribution of offensive content. Always communicate respectfully and sensitively online.

Compliance: Adhere to all relevant laws and regulations concerning internet and email use, including copyright and data protection laws.

Monitoring: The company reserves the right to monitor internet and email usage for security and compliance purposes....



**Task 6: Construct a QA Bot that leverages the LangChain and LLM to answer questions**

**In [10]:**

from langchain.embeddings import HuggingFaceEmbeddings

from langchain\_community.vectorstores import Chroma

from langchain\_community.document\_loaders import PyPDFLoader

from langchain.chains import RetrievalQA

from langchain.text\_splitter import RecursiveCharacterTextSplitter

from langchain.llms import HuggingFacePipeline

from transformers import GPTNeoForCausalLM, GPT2TokenizerFast, pipeline

import torch

import gradio as gr

\# Loading the GPT-Neo model and tokenizer

model\_name = "EleutherAI/gpt-neo-125M"

tokenizer = GPT2TokenizerFast.from\_pretrained(model\_name)  # Loads the GPT-2 tokenizer

model = GPTNeoForCausalLM.from\_pretrained(model\_name)  # Loads the GPT-Neo model

\# Creating a pipeline for text generation

generator = pipeline(

`    `"text-generation",  # Defining the task as text generation

`    `model=model,  # Using the GPT-Neo model

`    `tokenizer=tokenizer,  # Using the GPT-2 tokenizer

`    `max\_new\_tokens=500,  # Setting the maximum number of new tokens to generate

`    `device=0 if torch.cuda.is\_available() else -1  # Use GPU if available, otherwise use CPU

)

\# Function to return the HuggingFacePipeline (for text generation)

def get\_llm():

`    `return HuggingFacePipeline(pipeline=generator)

\# Function to load a PDF document

def document\_loader(file):

`    `loader = PyPDFLoader(file.name)  # Initializing a PDF loader

`    `loaded\_document = loader.load()  # Loading the document

`    `return loaded\_document

\# Function to split the document into chunks

def text\_splitter(data):

`    `text\_splitter = RecursiveCharacterTextSplitter(

`        `chunk\_size=1000,  # Defining chunk size (1000 characters per chunk)

`        `chunk\_overlap=50,  # Setting an overlap of 50 characters between chunks

`        `length\_function=len,  # Using the length of the text to split

`    `)

`    `chunks = text\_splitter.split\_documents(data)  # Splitting the document into chunks

`    `return chunks

\# Function to initialize the HuggingFace embedding model

def huggingface\_embedding():

`    `embedding\_model = HuggingFaceEmbeddings(model\_name="sentence-transformers/all-MiniLM-L6-v2")  # Load the embedding model

`    `return embedding\_model

\# Function to create a vector database from document chunks

def vector\_database(chunks):

`    `embedding\_model = huggingface\_embedding()  # Get the embedding model

`    `vectordb = Chroma.from\_documents(chunks, embedding\_model)  # Creating a vector store with embeddings

`    `return vectordb

\# Function to initialize the retriever from the vector database

def retriever(file):

`    `splits = document\_loader(file)  # Load the document

`    `chunks = text\_splitter(splits)  # Split the document into chunks

`    `vectordb = vector\_database(chunks)  # Create a vector database

`    `retriever = vectordb.as\_retriever()  # Initialize the retriever

`    `return retriever

\# Function for question-answering based on a file and query

def retriever\_qa(file, query):

`    `llm = get\_llm()  # Get the language model for text generation

`    `retriever\_obj = retriever(file)  # Initialize the retriever

`    `qa = RetrievalQA.from\_chain\_type(

`        `llm=llm,  # Using the language model

`        `chain\_type="stuff",  # Use the "stuff" chain type for retrieval-based QA

`        `retriever=retriever\_obj,  # Pass the retriever to the QA model

`        `return\_source\_documents=False,  # Don't return the source documents

`    `)

`    `response = qa.run(query)  # Run the QA model with the query

`    `return response

\# Gradio interface for the RAG Chatbot

rag\_application = gr.Interface(

`    `fn=retriever\_qa,  # Function to call for QA

`    `allow\_flagging="never",  # Disabling flagging of content

`    `inputs=[

`        `gr.File(label="Upload PDF File", file\_count="single", file\_types=['.pdf'], type="filepath"),  # Input: PDF file

`        `gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")  # Input: Query text

`    `],

`    `outputs=gr.Textbox(label="Output"),  # Output: Answer to the query

`    `title="RAG Chatbot",  # Title of the application

`    `description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."  # Description of the app

)

\# Launch the Gradio app (sharing enabled)

rag\_application.launch(share=True)

tokenizer\_config.json:   0%|          | 0.00/727 [00:00<?, ?B/s]

vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]

merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]

tokenizer.json:   0%|          | 0.00/2.11M [00:00<?, ?B/s]

special\_tokens\_map.json:   0%|          | 0.00/357 [00:00<?, ?B/s]

config.json:   0%|          | 0.00/1.01k [00:00<?, ?B/s]

model.safetensors:   0%|          | 0.00/526M [00:00<?, ?B/s]

generation\_config.json:   0%|          | 0.00/119 [00:00<?, ?B/s]

Device set to use cuda:0

Colab notebook detected. To show errors in colab notebook, set debug=True in launch()

\* Running on public URL: https://8a0aca4ca4f2e035bb.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)

Exported with [runcell](https://www.runcell.dev/tool/ipynb-to-html) — convert notebooks to HTML or PDF anytime at runcell.dev. 
