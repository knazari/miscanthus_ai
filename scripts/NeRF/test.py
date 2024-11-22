import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# data = pd.read_csv("/home/kia/Kiyanoush/Github/miscanthus_ai/epoch_loss.csv")['Average Loss']
# plt.plot(data)
# plt.xlabel("Epoch")
# plt.ylabel("L2 loss")
# plt.show()


img_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF/images/frame_0253.png"
img = Image.open(img_path).convert("RGB")
img_np = np.asarray(img)
print(img_np.shape)
