import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)

mpDraw = mp.solutions.drawing_utils

blue, green, red = 0, 0, 255
# Изменяемые параметры для перемещения квадрата
square_size = 50
square_pos = [300, 150]

while True:
    success, image = cap.read()
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))  # Пропорционально меняем размер кадра
    image = cv2.flip(image, 1)

    # Рисуем квадрат
    x1, y1 = square_pos
    x2, y2 = x1 + square_size, y1 + square_size
    cv2.rectangle(image, (x1, y1), (x2, y2), (blue, green, red), 5)

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handIdx, handLms in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                if handIdx == 0 or handIdx == 1:  # Обрабатываем палец для каждой руки
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8:  # Индексный палец
                        # Если палец касается левой границы квадрата
                        if (x1 - 15 <= cx <= x1 + 15) and y1 <= cy <= y2:
                            square_pos[0] += 10
                            red = 0
                            green = 255
                        # Если палец касается верхней границы квадрата
                        elif (y1 - 15 <= cy <= y1 + 15) and x1 <= cx <= x2:
                            square_pos[1] += 10
                            red = 0
                            green = 255
                        # Если палец касается правой границы квадрата
                        elif (x2 + 15 >= cx >= x2) and y1 <= cy <= y2:
                            square_pos[0] -= 10
                            red = 0
                            green = 255
                        # Если палец касается нижней границы квадрата
                        elif (y2 + 15 >= cy >= y2) and x1 <= cx <= x2:
                            square_pos[1] -= 10
                            red = 0
                            green = 255
                        else:
                            red = 255
                            green = 0


            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow('Find hand', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # Освобождает ресурсы связанные с VideoCapture
cv2.destroyAllWindows()  # Закрывает все открытые ок
