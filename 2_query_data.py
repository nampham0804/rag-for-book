import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# THAY ĐỔI Ở ĐÂY: Import mô hình của Google thay vì OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Tải các biến môi trường từ file .env
load_dotenv()

# Định nghĩa các hằng số
DB_FAISS_PATH = "faiss_index"
# THAY ĐỔI Ở ĐÂY: Lấy Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def run_query():
    """
    Hàm này thực hiện các bước sau:
    1. Tải cơ sở dữ liệu vector đã được lưu.
    2. Khởi tạo mô hình ngôn ngữ (LLM) từ Google (Gemini).
    3. Tạo một chuỗi RetrievalQA để kết hợp retriever và LLM.
    4. Chạy vòng lặp hỏi-đáp với người dùng.
    """
    # 1. Tải Vector DB (không thay đổi)
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Đã tải thành công vector database.")

    # 2. THAY ĐỔI Ở ĐÂY: Khởi tạo LLM của Google
    # Sử dụng mô hình "gemini-1.5-pro" (phiên bản mới)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
    print("Đã khởi tạo mô hình ngôn ngữ Google Gemini.")

    # 3. Tạo chuỗi RetrievalQA với cải tiến toàn diện
    # Tạo prompt template thông minh có thể xử lý mọi loại câu hỏi
    prompt_template = """Bạn là một trợ lý AI thông minh chuyên về phân tích và trả lời câu hỏi về tài liệu.

    HƯỚNG DẪN XỬ LÝ CÂU HỎI:
    
    1. Nếu câu hỏi yêu cầu tóm tắt chương/sách: Tìm tất cả thông tin liên quan và tóm tắt có cấu trúc
    2. Nếu câu hỏi về khái niệm/định nghĩa: Giải thích rõ ràng và đưa ra ví dụ
    3. Nếu câu hỏi so sánh/đối chiếu: Phân tích điểm giống và khác nhau
    4. Nếu câu hỏi về ý kiến/quan điểm: Trình bày các quan điểm khác nhau từ tài liệu
    5. Nếu câu hỏi thực hành/ứng dụng: Đưa ra hướng dẫn cụ thể
    6. Nếu câu hỏi không tìm thấy thông tin: Nói rõ và đề xuất tìm kiếm khác
    
    YÊU CẦU TRẢ LỜI:
    - Trả lời đầy đủ, chính xác và có cấu trúc
    - Sử dụng thông tin từ tài liệu được cung cấp
    - Nếu thông tin không đủ, hãy nói rõ
    - Trả lời bằng tiếng Việt (trừ khi được yêu cầu khác)
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 10}),  # Tăng số lượng chunks tìm kiếm
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("Đã tạo chuỗi RAG. Sẵn sàng để trả lời câu hỏi.")
    print("--------------------------------------------------")

    # 4. Vòng lặp hỏi-đáp với xử lý lỗi
    while True:
        try:
            query = input("\nNhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ").strip()
            if not query:
                print("Vui lòng nhập câu hỏi!")
                continue
                
            if query.lower() == 'exit':
                print("Cảm ơn bạn đã sử dụng hệ thống!")
                break
            
            print(f"\nĐang xử lý câu hỏi: '{query}'...")
            result = qa_chain.invoke({"query": query})
            
            print("\n> Câu trả lời:")
            print(result["result"])
            
            print(f"\n> Thông tin kỹ thuật:")
            print(f"- Số đoạn văn bản đã sử dụng: {len(result['source_documents'])}")
            print(f"- Tổng độ dài context: {sum(len(doc.page_content) for doc in result['source_documents'])} ký tự")
            
            if len(result['source_documents']) > 0:
                print("\n> Các đoạn văn bản đã sử dụng:")
                for i, doc in enumerate(result["source_documents"], 1):
                    print(f"{i}. {doc.page_content[:200]}...")
            else:
                print("\n> Không tìm thấy thông tin liên quan trong tài liệu.")
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n\nĐã dừng chương trình.")
            break
        except Exception as e:
            print(f"\nCó lỗi xảy ra: {str(e)}")
            print("Vui lòng thử lại với câu hỏi khác.")
            print("="*50)

if __name__ == "__main__":
    run_query()