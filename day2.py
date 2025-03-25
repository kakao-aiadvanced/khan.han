import getpass
import os
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = getpass.getpass()
llm = ChatOpenAI(model="gpt-4o-mini")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={'k': 6}
)

question = "like prompt."

# Relevance
prompt1 = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a grader assessing relevance of a retireved document to a user question.
    If the document contains keywords related to the user question, Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'relevance' and no premable or explanation.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Here is the rerieved document: \n\n {document} \n\n
    Here is the user question: {question} \n<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

docs = retriever.invoke(question)
doc_txt = docs[1].page_content

retrieval_grade = prompt1 | llm | JsonOutputParser()
score = retrieval_grade.invoke({"question": question, "document": doc_txt})

# Answer
prompt2 = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt2
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke(question)

# Hallucination
prompt3 = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    Evaluator detected a hallucination in the answer.
    If Evaluator detected a hallucination, Give a binary score 'yes' or 'no' as a JSON with a key 'hallucination'.
    Provide the binary score as a JSON with a single key 'hallucination' and no premable or explanation.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Here is the answer: {answer} \n<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["answer"],
)

hallucination_grade = prompt3 | llm | JsonOutputParser()

if 'yes' in score['relevance']:
    hallucination = hallucination_grade.invoke(answer)
    if 'yes' in hallucination['hallucination']:
        print('answer: hallucination is yes', answer)
    else:
        print('answer:', answer)
else:
    print('answer: relevance is no')
