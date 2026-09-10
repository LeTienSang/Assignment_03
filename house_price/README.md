# Vietnam Housing Price Prediction 2024

## Chạy backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python notebook/train.py
python -m uvicorn api.main:app --reload --port 8002
```

API: `POST http://localhost:8002/predict/house-price`  
Swagger: `http://localhost:8002/docs`

Notebook `notebook/train.py` so sánh Random Forest, Extra Trees, Gradient Boosting và DNN 5 tầng `220 -> 128 -> 64 -> 32 -> 16 -> 1` với BatchNorm, ReLU, Dropout 0.3 và MSE. Kết quả lưu tại `model/comparison.csv`; trọng số DL tại `model/model_5l.pt`, preprocessor chỉ fit trên train tại `model/preprocessor.joblib`.

## Chạy mobile

```powershell
cd mobile
npm start
```

Nhấn `a` để chạy Android Emulator. App gọi API qua `http://10.0.2.2:8002`.
Khi dùng điện thoại thật, đổi `API_URL` trong `mobile/App.tsx` sang IP LAN của máy tính.
