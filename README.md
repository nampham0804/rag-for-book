# Hệ Thống Hỏi Đáp - Hiểu Về Trái Tim

Ứng dụng web Flask cho phép người dùng đặt câu hỏi và nhận câu trả lời dựa trên nội dung sách "Hiểu Về Trái Tim" của tác giả Minh Niệm. Hệ thống sử dụng FAISS vector database để tìm kiếm semantic thông minh.

## Tính năng

- 🎨 Giao diện đẹp và hiện đại với thiết kế responsive
- 💬 Chat interface thân thiện với người dùng
- 🔍 Tìm kiếm semantic thông minh với FAISS vector database
- 💡 Gợi ý câu hỏi mẫu
- ⚡ Phản hồi nhanh chóng
- 📱 Tương thích với thiết bị di động
- 🔧 Panel quản lý dữ liệu tích hợp
- 📊 Hiển thị trạng thái hệ thống real-time
- 🔄 Tự động cập nhật database khi có thay đổi

## Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Chạy ứng dụng:**
```bash
python app.py
```

3. **Truy cập ứng dụng:**
Mở trình duyệt và truy cập: `http://localhost:5000`

## Cấu trúc dự án

```
├── app.py                 # File chính của ứng dụng Flask
├── data.txt              # Dữ liệu nội dung sách
├── 1_ingest_data.py     # Script tạo vector database
├── requirements.txt      # Danh sách dependencies
├── faiss_index/         # Thư mục chứa vector database
├── templates/
│   └── index.html       # Giao diện HTML
└── README.md            # Hướng dẫn sử dụng
```

## API Endpoints

- `GET /` - Trang chủ với giao diện chat
- `POST /search` - API tìm kiếm và trả lời câu hỏi
- `GET /api/suggestions` - API lấy danh sách câu hỏi gợi ý
- `GET /api/status` - API kiểm tra trạng thái hệ thống
- `POST /api/update-database` - API cập nhật vector database

## Cách sử dụng

### Giao diện chính
1. **Đặt câu hỏi:** Nhập câu hỏi vào ô input và nhấn Enter hoặc click nút "Gửi"
2. **Sử dụng gợi ý:** Click vào các câu hỏi gợi ý để tự động điền và gửi
3. **Xem lịch sử:** Cuộc trò chuyện được lưu lại trong phiên làm việc

### Panel quản lý (Sidebar)
1. **Trạng thái hệ thống:** Hiển thị thông tin về file dữ liệu, vector database và trạng thái hệ thống
2. **Quản lý dữ liệu:** 
   - Nút "Cập nhật Database" để chạy lại script ingest_data
   - Nút "Kiểm tra trạng thái" để refresh thông tin hệ thống
3. **Câu hỏi gợi ý:** Danh sách các câu hỏi mẫu để tham khảo

## Ví dụ câu hỏi

- "Tác giả Minh Niệm là ai?"
- "Nội dung sách Hiểu Về Trái Tim"
- "Làm thế nào để sống hạnh phúc?"
- "Cách chữa lành tổn thương tâm hồn"
- "Nghệ thuật sống hạnh phúc"
- "Tình yêu và hạnh phúc"
- "Chuyển hóa phiền não"
- "Hiểu về cảm xúc"
- "Sống trong thương yêu"
- "Khám phá nội tâm"

## Công nghệ sử dụng

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **UI/UX:** Font Awesome, Google Fonts
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Text Processing:** LangChain, RecursiveCharacterTextSplitter

## Tính năng nâng cao

### Tự động cập nhật
- Hệ thống tự động kiểm tra thay đổi trong file `data.txt`
- Tự động chạy `1_ingest_data.py` khi phát hiện thay đổi
- Tạo vector database mới với embeddings semantic

### Tìm kiếm thông minh
- Sử dụng FAISS vector database cho tìm kiếm semantic
- Chia văn bản thành chunks với overlap để không mất thông tin
- Tìm kiếm dựa trên ý nghĩa thay vì từ khóa chính xác

### Giao diện quản lý
- Panel trạng thái real-time
- Nút cập nhật database tích hợp
- Hiển thị thông tin chi tiết về hệ thống

## Phát triển

Để phát triển thêm tính năng:

1. **Thêm AI/ML:** Tích hợp với các model AI khác để cải thiện chất lượng câu trả lời
2. **Database:** Thêm database để lưu trữ lịch sử chat
3. **Authentication:** Thêm hệ thống đăng nhập
4. **Multi-language:** Hỗ trợ nhiều ngôn ngữ
5. **File Upload:** Cho phép upload file dữ liệu mới
6. **Export/Import:** Tính năng xuất/nhập dữ liệu

## Troubleshooting

### Lỗi thường gặp
1. **Vector database không tồn tại:** Click "Cập nhật Database" để tạo mới
2. **Lỗi khi load embeddings:** Kiểm tra kết nối internet để download model
3. **File data.txt không tồn tại:** Đảm bảo file dữ liệu có trong thư mục

### Logs
- Kiểm tra console để xem thông tin chi tiết về quá trình khởi động
- Các thông báo lỗi sẽ hiển thị trong chat interface

## License

MIT License 