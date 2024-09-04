import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# tiktoken 설정
tokenizer = tiktoken.get_encoding("cl100k_base")

# tiktoken을 사용하여 텍스트의 토큰 길이를 계산
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def process_pdf(file_path):
    try:
        # PDF 로드 및 텍스트 추출
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # RecursiveCharacterTextSplitter를 사용한 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Chunk 크기 (토큰 기준)
            chunk_overlap=200,  # Overlap 크기 (토큰 기준)
            length_function=tiktoken_len  # tiktoken 기반 텍스트 길이 함수 사용
        )

        # 문서 텍스트 분할
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    except Exception as e:
        logging.error(f"PDF 처리 중 오류 발생: {e}")
        raise

