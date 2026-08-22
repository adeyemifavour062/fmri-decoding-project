from nilearn import image, datasets
from nilearn.image import index_img
import numpy as np
haxby = datasets.fetch_haxby()
fmri_img = image.load_img(haxby.func[0])
print(fmri_img)
import pandas as pd
labels = pd.read_csv(haxby.session_target[0], sep=" ")
print(labels.head())
mask = labels['labels'].isin(['face','house'])
filtered_labels = labels[mask]
print(filtered_labels['labels'].value_counts())
indices = np.where(mask)[0]
fmri_img_filtered = index_img(fmri_img, indices)
print(fmri_img_filtered)
from nilearn.decoding import Decoder
decoder = Decoder(estimator='svc',mask=haxby.mask,standardize=True)
decoder.fit(fmri_img_filtered, filtered_labels['labels'])
print("Training complete!")
scores = decoder.cv_scores_
for label, values in scores.items():
    print(label, np.mean(values))
