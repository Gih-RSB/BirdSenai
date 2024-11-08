import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import random

# Inicialização do mixer do pygame
pygame.mixer.init()

# Arquivos de áudio para os sons
audio_files = ["bemtevi.mp3", "carcara.mp3", "joaodebarro.mp3", "sabia.mp3", "seriema.mp3"]
alarme_audio = "alarme.mp3"  # Caminho para o áudio do alarme

audio_nomes = {
    "bemtevi.mp3": "bem-te-vi",
    "carcara.mp3": "carcara",
    "joaodebarro.mp3": "joao-de-barro",
    "sabia.mp3": "sabia",
    "seriema.mp3": "seriema"
}

p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Limiar para EAR (eyes aspect ratio) e MAR (mouth aspect ratio)
ear_limiar = 0.27
mar_limiar_bocejo = 0.3  # Limiar para detectar o bocejo
dormindo = 0
cap = cv2.VideoCapture(0)

# Inicializando a biblioteca MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
som_tocando = False
ultimo_tempo_audio = time.time()

# Flag para saber se o alarme está tocando
alarme_tocando = False

# Cor do efeito arco-íris (inicializada)
cor_arcoiris = (255, 105, 180)  # Inicial com rosa claro
frame_counter = 0  # Contador de frames para o efeito arco-íris

# Carregar imagem do carrinho e redimensionar
carrinho_icon = cv2.imread('carrinho.png')  # Substitua 'carrinho.png' pelo caminho da imagem
carrinho_icon = cv2.resize(carrinho_icon, (50, 50))  # Redimensionar o ícone do carrinho

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Aplicar filtro de nitidez para melhorar a qualidade da câmera
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Atualizar cor para efeito arco-íris a cada 5 frames para o rosto
        if frame_counter % 5 == 0:
            cor_arcoiris = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        frame_counter += 1

        if saida_facemesh.multi_face_landmarks:
            print("Rosto detectado")
            tempo_atual = time.time()
            if tempo_atual - ultimo_tempo_audio >= 10 and som_tocando == False:
                audio_aleatorio = random.choice(audio_files)
                try:
                    pygame.mixer.music.load(audio_aleatorio)
                    pygame.mixer.music.play()
                    ultimo_tempo_audio = tempo_atual
                    nome_audio = audio_nomes.get(audio_aleatorio, "desconhecido")
                except pygame.error as e:
                    print(f"Erro ao carregar o áudio {audio_aleatorio}: {e}")
        else:
            print("Nenhum rosto detectado")
            pygame.mixer.music.stop()  
            som_tocando = False  

        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                # Alterar apenas a cor da máscara do rosto (não toda a imagem)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=cor_arcoiris, thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=cor_arcoiris, thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                # Cálculo do EAR (estado dos olhos)
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, cor_arcoiris, 2)

                # Exibindo o nome do áudio
                cv2.putText(frame, f"Audio: {nome_audio}", (1, 110),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, cor_arcoiris, 2)

                # Cálculo do MAR (estado da boca)
                mar = calculo_mar(face, p_boca)
                mar_status = "Boca Aberta" if mar >= mar_limiar_bocejo else "Boca Fechada"
                mar_color = cor_arcoiris if mar >= mar_limiar_bocejo else (255, 0, 0)

                # Exibindo o status da boca
                cv2.putText(frame, f"{mar_status}: {round(mar, 2)}", (1, 50),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1, mar_color, 2, cv2.LINE_AA)

                # Verificação do estado de sono (dormindo ou acordado)
                estado_sono = "Dormindo" if ear < ear_limiar else "Acordada"
                estado_sono_color = cor_arcoiris  

                # Exibindo o status de sono
                cv2.putText(frame, f"Status: {estado_sono}", (1, 80),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1, estado_sono_color, 2, cv2.LINE_AA)

                # Verificar se a boca está aberta o suficiente para detectar bocejo e tocar alarme
                if mar >= mar_limiar_bocejo and not alarme_tocando:
                    try:
                        pygame.mixer.music.load(alarme_audio)
                        pygame.mixer.music.play(-1)
                        alarme_tocando = True
                    except pygame.error as e:
                        print(f"Erro ao carregar o áudio do alarme: {e}")
                elif mar < mar_limiar_bocejo and alarme_tocando:
                    pygame.mixer.music.stop()  
                    alarme_tocando = False

        except Exception as e:
            print("Erro:", e)

        finally:
            print("Processamento concluído")

        # Exibir ícone do carrinho no canto superior direito
        frame[10:60, largura-60:largura-10] = carrinho_icon

        # Exibindo a imagem final
        cv2.imshow('Camera', frame)

        # Condição para sair do loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
