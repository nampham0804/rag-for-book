from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
import re
import subprocess
import sys
from pathlib import Path

# Import các thư viện cho vector search
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

app = Flask(__name__)

# Tải các biến môi trường từ file .env
load_dotenv()

# Định nghĩa các hằng số
DATA_PATH = "data.txt"
DB_FAISS_PATH = "faiss_index"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def run_ingest_data():
    """Chạy file 1_ingest_data.py để cập nhật vector database"""
    try:
        print("🔄 Đang cập nhật vector database...")
        result = subprocess.run([sys.executable, "1_ingest_data.py"], 
                              capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print("✅ Cập nhật vector database thành công!")
            return True
        else:
            print(f"❌ Lỗi khi cập nhật vector database: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi chạy ingest_data: {e}")
        return False

def check_and_update_database():
    """Kiểm tra và cập nhật database nếu cần"""
    data_file = Path(DATA_PATH)
    db_path = Path(DB_FAISS_PATH)
    
    # Kiểm tra xem có cần cập nhật không
    if not db_path.exists() or not (db_path / "index.faiss").exists():
        print("📊 Vector database chưa tồn tại, đang tạo mới...")
        return run_ingest_data()
    
    # Kiểm tra xem file data.txt có thay đổi không
    if data_file.exists():
        data_mtime = data_file.stat().st_mtime
        db_mtime = db_path.stat().st_mtime if db_path.exists() else 0
        
        if data_mtime > db_mtime:
            print("📝 Phát hiện thay đổi trong data.txt, đang cập nhật database...")
            return run_ingest_data()
    
    return True

def load_vector_database():
    """Load vector database từ FAISS"""
    try:
        if not os.path.exists(DB_FAISS_PATH):
            print("❌ Vector database không tồn tại!")
            return None
        
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        
        # Thêm allow_dangerous_deserialization=True để cho phép load database
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ Đã load vector database thành công!")
        return db
    except Exception as e:
        print(f"❌ Lỗi khi load vector database: {e}")
        return None

def create_qa_chain(db):
    """Tạo QA chain với Google Gemini LLM"""
    try:
        if not GOOGLE_API_KEY:
            print("❌ Không tìm thấy GOOGLE_API_KEY trong file .env")
            return None
        
        # Khởi tạo LLM của Google
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.2, 
            google_api_key=GOOGLE_API_KEY
        )
        
        # Tạo prompt template thông minh
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
        
        # Tạo QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 10}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        print(f"❌ Lỗi khi tạo QA chain: {e}")
        return None

def generate_answer_with_llm(query, qa_chain):
    """Tạo câu trả lời sử dụng LLM"""
    if not qa_chain:
        return "Xin lỗi, không thể khởi tạo AI model. Vui lòng kiểm tra API key và thử lại."
    
    try:
        # Sử dụng QA chain để tạo câu trả lời
        result = qa_chain.invoke({"query": query})
        
        # Lấy câu trả lời từ LLM
        answer = result["result"]
        
        # Thêm thông tin về số đoạn văn bản đã sử dụng
        if result.get('source_documents'):
            num_docs = len(result['source_documents'])
            total_chars = sum(len(doc.page_content) for doc in result['source_documents'])
            answer += f"\n\n---\n*Sử dụng {num_docs} đoạn văn bản ({total_chars} ký tự)*"
        
        return answer
    except Exception as e:
        print(f"❌ Lỗi khi tạo câu trả lời: {e}")
        return f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Vui lòng nhập câu hỏi'})
        
        # Load vector database
        db = load_vector_database()
        if not db:
            return jsonify({'error': 'Không thể load vector database. Vui lòng kiểm tra lại.'})
        
        # Tạo QA chain với LLM
        qa_chain = create_qa_chain(db)
        if not qa_chain:
            return jsonify({'error': 'Không thể khởi tạo AI model. Vui lòng kiểm tra API key.'})
        
        # Tạo câu trả lời sử dụng LLM
        answer = generate_answer_with_llm(query, qa_chain)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'})

@app.route('/api/suggestions')
def get_suggestions():
    """API lấy danh sách câu hỏi gợi ý"""
    suggestions = [
        "Tác giả Minh Niệm là ai?",
        "Nội dung sách Hiểu Về Trái Tim",
        "Làm thế nào để sống hạnh phúc?",
        "Cách chữa lành tổn thương tâm hồn",
        "Nghệ thuật sống hạnh phúc",
        "Tình yêu và hạnh phúc",
        "Chuyển hóa phiền não",
        "Hiểu về cảm xúc",
        "Sống trong thương yêu",
        "Khám phá nội tâm",
        "Cách đối mặt với khó khăn",
        "Tìm kiếm hạnh phúc chân thật",
        "Phát triển tâm linh",
        "Cách yêu thương bản thân",
        "Xây dựng mối quan hệ tốt đẹp"
    ]
    return jsonify({'suggestions': suggestions})

@app.route('/api/status')
def get_status():
    """API kiểm tra trạng thái hệ thống"""
    try:
        data_exists = os.path.exists(DATA_PATH)
        db_exists = os.path.exists(DB_FAISS_PATH)
        
        status = {
            'data_file_exists': data_exists,
            'vector_db_exists': db_exists,
            'system_ready': data_exists and db_exists,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if data_exists:
            data_size = os.path.getsize(DATA_PATH)
            status['data_size_mb'] = round(data_size / (1024 * 1024), 2)
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/update-database', methods=['POST'])
def update_database():
    """API cập nhật vector database"""
    try:
        success = run_ingest_data()
        if success:
            return jsonify({
                'success': True,
                'message': 'Cập nhật vector database thành công!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Có lỗi xảy ra khi cập nhật database.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        })

if __name__ == '__main__':
    print("🚀 Khởi động hệ thống hỏi đáp...")
    
    # Kiểm tra và cập nhật database khi khởi động
    if check_and_update_database():
        print("✅ Hệ thống sẵn sàng!")
    else:
        print("⚠️ Có lỗi khi khởi tạo database, nhưng ứng dụng vẫn có thể chạy.")
    
    print("🌐 Khởi động web server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 