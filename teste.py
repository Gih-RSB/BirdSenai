import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: A câmera não foi encontrada.")
else:
    print("Câmera encontrada e aberta com sucesso.")
    cap.release()

