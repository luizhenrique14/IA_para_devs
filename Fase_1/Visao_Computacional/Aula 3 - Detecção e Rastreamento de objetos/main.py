import cv2

# Carrega o modelo pré-treinado MobileNet SSD
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',  # Estrutura da rede
    'mobilenet_iter_73000.caffemodel'  # Pesos treinados
)

# Inicializa a câmera (use 0 para webcam)
cap = cv2.VideoCapture(0)

# Inicializa variáveis de rastreamento
tracker = None
initBB = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Se ainda não tem objeto para rastrear, detecta
    if initBB is None:
        # Prepara imagem para o modelo
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Procura por objetos com confiança alta
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Converte coordenadas
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Define a região para rastrear
                initBB = (startX, startY, endX - startX, endY - startY)

                # Cria e inicializa o rastreador (KCF)
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, initBB)
                break
    else:
        # Atualiza rastreamento
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Rastreando...", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Objeto perdido", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            initBB = None  # Reinicia detecção

    # Mostra o frame na tela
    cv2.imshow("Detecção e Rastreamento", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera a câmera e fecha janelas
cap.release()
cv2.destroyAllWindows()
