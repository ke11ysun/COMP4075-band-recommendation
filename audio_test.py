import audioBasicIO as aio
import audioFeatureExtraction as afe
import numpy as np
import os
from sklearn.decomposition import PCA

all_bands = []
all_mtf = []

proj_path = '/users/sunjingxuan/desktop/project_songs'
for band_folder in enumerate(os.listdir(proj_path)):
    band_path = os.path.join(proj_path,band_folder[1])
    all_bands.append(band_path)
    # print(band_path)

for band in all_bands:
    dirName = band
    print(dirName)
    [allMtFeatures, wavFilesList2] = afe.dirWavFeatureExtraction(dirName, 1.0, 1.0, 0.050, 0.050, False)
    print(wavFilesList2)
    print(allMtFeatures.shape)
    # print(allMtFeatures)

    pca = PCA(n_components=10)
    pcaf = pca.fit_transform(allMtFeatures.T).T
    print(pcaf.shape)

    np.save('/Users/sunjingxuan/desktop/midft_' + band.split("/")[-1] + '.npy', pcaf)




