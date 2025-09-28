# ProjetReconnaissanceVisage : Détection des visages avec OpenCV et HaarCascade
L'objectif de ce projet est d'utiliser la cascade de Haar pour faire de la détéction de visages, pour cela on aura besoin de télécharger le fichier : haarcascade_frontalface_default.xml ; Celui ci est un modèle déjà entrainé à détecter des motifs en noir et blanc qui définissent le contraste d'un visage. C'est aussi pourquoi la détéction fonctionnera moins bien sur des visages plus foncées ou avec moins de contraste entre les différentes parties du visage.

Je commence par tester ce modèle sur une photo : 

```
import cv2
face_cascade = cv2.CascadeClassifier(r"C:\Users\Abdessamad\Downloads\haarcascade_frontalface_default.xml")
img = cv2.imread(r"C:\Users\Abdessamad\Downloads\test.png") #lecture de l'image en entrée
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Changement de l'image en niveaux de gris
faces = face_cascade.detectMultiScale(gray, 1.1 , 4) #Détection du visage
for (x, y, w , h) in faces:
    cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0) ,3)  #Dessin du réctangle
cv2.imshow('img', img) #Affichage de l'image
cv2.waitKey()

```
Et ça donne ça : 

![Exercice3duTP](CaptureExo1.png)
Comme on peut le voir, la cascade de Haar a bien reconnu mon visage même si la qualité de l'image laisse a désirer...

On refait la même chose sur une vidéo pour voir le résultat : 

```
import cv2
face_cascade = cv2.CascadeClassifier(r"C:\Users\Abdessamad\Downloads\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(r"C:\Users\Abdessamad\Downloads\testvid.mp4")#Lecture de la vidéo en entrée
while cap.isOpened(): 
    _, img = cap.read() #Boucle sur chaque frame de la vidéo
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Changement de la frame en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, 1.1 , 4) #Détéction du visage
    for (x, y, w , h) in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0) ,3) #Dessin du réctangle sur chaque frame
        cv2.imshow('img', img) #affichage de la frame
        if cv2.waitKey(1) & 0xFF == ord('q'): #Si appui sur q, la rediffusion s'arrête 
            break
cap.release()
```

Et voici ce que ça donne en vidéo : 
![Exercice4duTP](TestVidExo4.mp4)

Maintenant que nous avons fait ça, on nous propose dans le TP de faire aussi la détéction des yeux. J'ai décidé de combiner ça avec le fait d'utiliser une image, une vidéo ou la webcam pour tester le modèle sur un seul code. 

Le voici : 
```
import cv2
import argparse

# ---------- Détection et annotation ----------
def process_frame(frame, face_cascade, eye_cascade):
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
    )

    out = frame.copy()
    for (x, y, w, h) in faces:
        # Rectangle BLEU autour du visage
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ROI visage pour chercher les yeux
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = out[y:y + h, x:x + w]

        # Détection des yeux dans le visage
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
        )
        for (ex, ey, ew, eh) in eyes:
            # Rectangle VERT autour de chaque œil (coords locales au visage)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return out

# ---------- Modes ----------
def run_image(path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if face_cascade.empty() or eye_cascade.empty():
        raise FileNotFoundError("Impossible de charger les Haar cascades (OpenCV).")

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")

    annotated = process_frame(img, face_cascade, eye_cascade)
    cv2.imshow("Visage (bleu) + Yeux (vert)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_stream(src=0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if face_cascade.empty() or eye_cascade.empty():
        raise FileNotFoundError("Impossible de charger les Haar cascades (OpenCV).")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra/vidéo.")

    print("Appuie sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_frame(frame, face_cascade, eye_cascade)
        if annotated is None:
            break
        cv2.imshow("Visage (bleu) + Yeux (vert)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- Main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détection visage/yeux avec OpenCV (Haar Cascades)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--webcam", action="store_true", help="Utiliser la webcam")
    grp.add_argument("--image", type=str, help="Chemin image")
    grp.add_argument("--video", type=str, help="Chemin vidéo")
    args = ap.parse_args()

    if args.webcam:
        run_stream(0)
    elif args.image:
        run_image(args.image)
    else:
        run_stream(args.video)

```
Ici on ajoute la bibliothéque argparse pour pouvoir lancer le code python avec différents arguments ! Copiez-collez le programme, testez avec l'argument --webcam si vous en avez une, --image "chemin" ou --video "chemin". 

Aprés test, je remarque que le fait d'avoir des lunettes fait que la détéction des yeux est assez mauvaise. 
