# Diabetes Prediction

## Chạy backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python notebook/train.py
python -m uvicorn api.main:app --reload --port 8001
```

API: `POST http://localhost:8001/predict/diabetes`  
Swagger: `http://localhost:8001/docs`

Notebook `notebook/train.py` huấn luyện ba baseline và DNN 5 tầng `8 -> 64 -> 32 -> 16 -> 8 -> 1` với BatchNorm, ReLU, Dropout 0.3 và Sigmoid/BCE. Kết quả so sánh được lưu tại `model/comparison.csv`; trọng số tại `model/model_5l.pt` và preprocessor chỉ được fit trên tập train tại `model/preprocessor.joblib`.

## Chạy mobile

```powershell
cd mobile
npm start
```

Nhấn `a` để chạy Android Emulator. App gọi API qua `http://10.0.2.2:8001`.
Khi dùng điện thoại thật, đổi `API_URL` trong `mobile/App.tsx` sang IP LAN của máy tính.
