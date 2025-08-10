import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Định nghĩa các hằng số
DATA_PATH = "data.txt"
DB_FAISS_PATH = "faiss_index"

def create_vector_db():
    """
    Hàm này thực hiện các bước sau:
    1. Tải tài liệu từ file data.txt.
    2. Chia tài liệu thành các đoạn nhỏ hơn (chunks).
    3. Sử dụng mô hình embedding để chuyển các đoạn văn bản thành vector.
    4. Tạo một cơ sở dữ liệu vector từ các embeddings và lưu trữ cục bộ.
    """
    # 1. Tải tài liệu
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()
    print(f"Đã tải thành công {len(documents)} tài liệu.")

    # 2. Chia tài liệu thành các đoạn nhỏ với cải tiến
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tăng chunk size để giữ được nhiều nội dung hơn
        chunk_overlap=200,  # Tăng overlap để không mất thông tin
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Cải thiện cách phân đoạn
    )
    texts = text_splitter.split_documents(documents)
    print(f"Đã chia tài liệu thành {len(texts)} đoạn văn bản.")
    print(f"Chunk size: 1000, Overlap: 200")

    # 3. Tạo embeddings
    # Chúng ta sẽ sử dụng một mô hình embedding miễn phí từ HuggingFace
    # Mô hình này sẽ chạy trên máy của bạn
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} # Chạy trên CPU
    )
    print("Đã khởi tạo mô hình embedding.")

    # 4. Tạo và lưu trữ Vector DB
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Đã tạo và lưu trữ thành công vector database tại '{DB_FAISS_PATH}'.")
    
    # 5. Hiển thị thống kê
    print(f"\nThống kê:")
    print(f"- Tổng số chunks: {len(texts)}")
    print(f"- Kích thước chunk trung bình: {sum(len(text.page_content) for text in texts) // len(texts)} ký tự")
    print(f"- Chunk nhỏ nhất: {min(len(text.page_content) for text in texts)} ký tự")
    print(f"- Chunk lớn nhất: {max(len(text.page_content) for text in texts)} ký tự")

if __name__ == "__main__":
    create_vector_db()