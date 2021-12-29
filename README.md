#   Compress image using K-Means Clustering
    
    Chương trình giảm dung lượng ảnh, phân cụm theo màu sắc
    
#   1. Yêu cầu cài đặt đối với chương trình

    Tải và cài đặt Python: https://www.python.org/downloads/

    Mở terminal, tiến hành cài đặt:
        
        PyQt5:      pip install PyQt5

        Pillow:     pip install Pillow

        NumPy:      pip install numpy

        Panda:      pip install panda

        Sklearn:    pip install sklearn

#   2. Khởi chạy chương trình:

    B1: Mở terminal, gõ "cd {path}" và ấn enter, với {path} là đường dẫn tới thư mục chứa file "GUI.py"

    B2: Gõ "python GUI.py" và ấn enter

#   3. Giải thích các nút và nhãn trong giao diện ứng dụng:

    "Open image": Chọn ảnh từ máy để đưa vào xử lý

    "Export image": Lưu ảnh đã qua xử lý về máy

    "Show original image": Mở hình ảnh gốc trong trình xem ảnh

    "Show compressed image": Mở hình ảnh đã qua xử lý trong trình xem ảnh

    "Optimal K (base on exp_var)": Tìm K tối ưu theo Explained variance (Từ k = OptimalK trở đi, tốc độ tăng Explained variance theo k chậm lại đáng kể)

    "Custom K": Chọn K để xử lý

    "Max iteration": Số iteration tối đa

    "Execute": Bắt đầu xử lý

    "Number of colors": Số màu sắc xuất hiện trong bức ảnh trước và sau khi xử lý

    "Image size": Kích thước hình ảnh trước và sau khi xử lý

    "Size different": Độ giảm kích thước của ảnh sau khi xử lý, đơn vị %

    "Explained variance": Phương sai giải thích, cho biết mức độ biểu diễn của hình ảnh sau khi xử lý so với ảnh gốc, đơn vị %
