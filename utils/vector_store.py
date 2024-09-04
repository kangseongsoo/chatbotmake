from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # 최신 패키지에서 ChatOpenAI 임포트
import os
import logging

# ChromaDB 저장 경로 설정
DB_DIR = "db"
os.makedirs(DB_DIR, exist_ok=True)  # db 디렉터리 생성 (이미 존재하면 생략)

# OpenAI Embeddings 초기화
embeddings = OpenAIEmbeddings()

# ChromaDB 벡터 스토어 초기화 (db 디렉터리 아래에 저장)
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory=DB_DIR  # 저장 경로를 db 디렉터리로 설정
)

def store_embeddings(documents):
    try:
        # 문서 벡터화 및 저장
        vector_store.add_documents(documents)
        logging.info(f"벡터 데이터가 {DB_DIR} 디렉터리에 저장되었습니다.")
    except Exception as e:
        logging.error(f"벡터 저장 중 오류 발생: {e}")
        raise

def query_vector_store(query_text):
    try:
        # 벡터 쿼리
        results = vector_store.similarity_search(query_text, k=1)
        logging.info(f"검색된 결과: {results}")
        
        if results:
            # LLM 호출
            llm = ChatOpenAI(temperature=0.7)
            response = llm.invoke([{"role": "user", "content": results[0].page_content}])
            return response.content  # 응답 내용 추출
        return "답변을 찾을 수 없습니다."
    except Exception as e:
        logging.error(f"벡터 검색 중 오류 발생: {e}")
        return "답변을 생성하는 중 오류가 발생했습니다."

