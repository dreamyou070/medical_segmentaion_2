# We will be using the make_blobs method
# in order to generate our own data.
import torch
from sklearn.cluster import DBSCAN

training_data_ima_embedding = torch.randn(1,176,768)
training_data_txt_embedding = torch.randn(1,3,768)

cls_token = training_data_ima_embedding[:,0,:]
image_embedding = training_data_ima_embedding[:,1:,:]

db = DBSCAN(eps=0.3, min_samples=10).fit(image_embedding)
labels = db.labels_
