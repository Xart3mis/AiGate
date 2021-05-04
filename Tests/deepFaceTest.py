from deepface import DeepFace
result  = DeepFace.verify("img1.jpg", "img2.jpg")
print("Is verified: ", result["verified"])