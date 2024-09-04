from flask import Flask, render_template, request, jsonify, redirect, url_for
from utils.pdf_processing import process_pdf
from utils.vector_store import store_embeddings, query_vector_store
from werkzeug.utils import secure_filename
import os
import logging
import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 파일 업로드 및 학습
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':  # POST 요청 처리 부분
        pdf_file = request.files['file']  # HTML에서 file input으로 변경된 부분
        if pdf_file:
            # 안전한 파일 이름 처리
            filename = secure_filename(pdf_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 파일을 지정된 경로에 저장
            pdf_file.save(file_path)

            # PDF 파일 처리 및 벡터화
            try:
                documents = process_pdf(file_path)
                store_embeddings(documents)
                return jsonify({'message': 'PDF가 성공적으로 처리되고 학습되었습니다!'})  # JSON 응답 반환
            except Exception as e:
                logging.error(f"PDF 처리 중 오류 발생: {e}")
                return jsonify({'message': 'PDF 처리 중 오류가 발생했습니다.'}), 500

    # GET 요청 처리 - 파일 목록 보여줌
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        files.append({
            'filename': filename,
            'upload_date': datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        })

    return render_template('data_train.html', files=files)  # HTML 페이지 반환


# 파일 삭제
@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(f"파일 삭제됨: {filename}")
        return redirect(url_for('train'))
    return jsonify({'message': '파일을 찾을 수 없습니다.'}), 404

# 챗봇 기능
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    try:
        if request.method == 'POST':
            user_question = request.json.get('message')
            if not user_question:
                return jsonify({'answer': '질문이 제공되지 않았습니다.'}), 400

            answer = query_vector_store(user_question)
            return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f"챗봇 응답 중 오류 발생: {e}", exc_info=True)
        return jsonify({'answer': '답변을 생성하는 중 오류가 발생했습니다.'}), 500

    return render_template('chatbot.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)

