import os
import numpy as np
import imageio
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define the directory containing the face images
image_dir = r"C:\Users\Zakaria\Desktop\att_faces-#2"

# Initialize lists to hold the images and labels
images = []
labels = []

# Set the number of subjects to include
num_subjects = 5

# Loop through the first `num_subjects` subject directories
subject_dirs = sorted(os.listdir(image_dir))[:num_subjects]
for subject_dir in subject_dirs:
    subject_path = os.path.join(image_dir, subject_dir)
    if os.path.isdir(subject_path):
        label = int(subject_dir[1:]) - 1  # Assuming directories are named as 'sX'
        # Loop through each image in the subject directory
        for image_name in os.listdir(subject_path):
            if image_name.endswith('.pgm'):
                image_path = os.path.join(subject_path, image_name)
                image = imageio.imread(image_path)
                images.append(image)
                labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Reshape images for the model
n_samples, h, w = images.shape
X = images.reshape(n_samples, h * w)
y = labels

print(f"Loaded {n_samples} images with shape {h}x{w}")

# Split the dataset into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the PCA and SVM pipeline
# Reduce the number of PCA components
n_components = min(n_samples, h * w, 20)  # Adjusted to 20 to ensure it fits within the sample size
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized', random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# Define parameter grid for GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

# Measure execution time
import time
start_time = time.time()
grid.fit(Xtrain, ytrain)
end_time = time.time()

print(f"GridSearchCV took {end_time - start_time:.2f} seconds")
print(grid.best_params_)

# Predict and visualize
model = grid.best_estimator_
yfit = model.predict(Xtest)

# Plot some predictions
n_plots = min(24, len(Xtest))  # Plot at most 24 images or the number of images in the test set
fig, ax = plt.subplots(4, 6, figsize=(12, 8))
for i, axi in enumerate(ax.flat):
    if i < n_plots:
        axi.imshow(Xtest[i].reshape(h, w), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(f"Predicted: {yfit[i]}\nTrue: {ytest[i]}", color='black' if yfit[i] == ytest[i] else 'red')
    else:
        axi.axis('off')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()

# Classification report
print(classification_report(ytest, yfit))

# Confusion matrix
mat = confusion_matrix(ytest, yfit)
plt.figure(figsize=(10, 8))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=np.arange(1, num_subjects + 1),  # Adjust labels to number of subjects
            yticklabels=np.arange(1, num_subjects + 1))
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
