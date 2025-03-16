from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import cvzone
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Configurações
class Config:
    NOTIFICATION_THRESHOLD = 5  # Notificar quando houver menos vagas que isso

# Classe para gerenciar dados do estacionamento
class ParkingManager:
    def __init__(self):
        self.history = []  # Mantenha o histórico em memória
        self.current_spaces = 0
        self.total_spaces = 0

    def update_stats(self, free_spaces, total_spaces):
        self.current_spaces = free_spaces
        self.total_spaces = total_spaces
        
        # Registrar histórico
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({
            "timestamp": timestamp,
            "free_spaces": free_spaces,
            "total_spaces": total_spaces
        })
        
        # Manter apenas as últimas 1000 entradas
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def get_stats(self):
        if self.total_spaces == 0:  # Para evitar divisão por zero
            return {"current": 0, "total": 0, "percentage": 0}
        
        return {
            "current": self.current_spaces,
            "total": self.total_spaces,
            "percentage": round((self.current_spaces / self.total_spaces) * 100, 1)
        }

parking_manager = ParkingManager()

def load_car_park_positions():
    try:
        with open('CarParkPos', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("CarParkPos file not found. Please make sure the file is in the same directory.")
        return []

posList = load_car_park_positions()
width, height = 107, 48

def checkParkingSpace(imgPro, img):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        
        # Definir cor e contagem de espaços livres
        if count < 900:
            color = (0, 255, 0)  # Verde para vagas livres
            spaceCounter += 1
            thickness = 2
        else:
            color = (0, 0, 255)  # Vermelho para vagas ocupadas
            thickness = 2
        
        # Desenhar retângulo com bordas arredondadas
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x + 10, y + height - 10), scale=0.8, 
                          thickness=1, offset=5, colorR=(255, 255, 255))
    
    # Atualizar estatísticas
    parking_manager.update_stats(spaceCounter, len(posList))  # Certifique-se de que está usando o total de vagas correto
    
    # Exibir informações na imagem
    cvzone.putTextRect(
        img, f'Vagas Livres: {spaceCounter}/{len(posList)}',
        (30, 50), scale=1.5, thickness=2, offset=10,
        colorR=(0, 200, 100), colorB=(50, 50, 50), border=5
    )
    
    # Adicionar timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cvzone.putTextRect(
        img, timestamp,
        (30, 100), scale=1, thickness=1, offset=5,
        colorR=(200, 200, 200), colorB=(50, 50, 50)
    )
    
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(parking_manager.get_stats())

def generate_frames():
    cap = cv2.VideoCapture('carPark.mp4')
    while True:
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Processamento de imagem
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
        
        img = checkParkingSpace(imgDilate, img)
        
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parking_manager.update_stats(15, 15)  # Inicializa o total de vagas se necessário
    app.run(debug=True)