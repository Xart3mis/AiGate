from sklearn import svm
import face_recognition
import pickle
import os

train_directory =     "/home/bighero/AiGate/Tests/face_rec/train_dir/"
model_save_path =     "/home/bighero/AiGate/Tests/face_rec/model/SVM_Classifier.pickle"
encodings_save_path = "/home/bighero/AiGate/Tests/face_rec/model/Encodings.pickle"

encodings = []
names = []

# Training directory
train_dir = os.listdir(train_directory)

# Loop through each person in the training directory
for person in train_dir:
	pix = os.listdir(train_directory + person)

	# Loop through each training image for the current person
	for person_img in pix:
		# Get the face encodings for the face in each image file
		face = face_recognition.load_image_file(train_directory + person + "/" + person_img)
		face_bounding_boxes = face_recognition.face_locations(face)

		#If training image contains exactly one face
		if len(face_bounding_boxes) == 1:
		    face_enc = face_recognition.face_encodings(face)[0]
		    # Add face encoding for current image with corresponding label (name) to the training data
		    encodings.append(face_enc)
		    names.append(person)
		    print(f"{pix.index(person_img)+1}/{len(pix)} for {person}")
		else:
		    print(person + "/" + person_img + " was skipped and can't be used for training")
	
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

if encodings_save_path is not None:
    with open(encodings_save_path, 'wb') as f:
        pickle.dump(encodings, f)
		    
if model_save_path is not None:
    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)
