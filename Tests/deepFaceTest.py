from deepface import DeepFace

models = ["VGG-Face", "Facenet", "OpenFace",
          "DeepFace", "DeepID", "ArcFace", "Dlib"]

for model in models:
    result = DeepFace.verify(
        "D:\\Documents\\dataset\\train\seif zayed\\20210401_002105.jpg",
        "D:\\Documents\\dataset\\train\\seif zayed\\20210401_002113.jpg", model_name=model)

    print(result["verified"])
