import matplotlib.pyplot as plt
import numpy as np

x= np.linspace(1,130, 130)

print(meanPixel.mean(axis=0))
plt.plot(x, meanPixel)
plt.title('Mean pixel accuracy by epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Pixel Accuracy')
plt.savefig('mpa.jpg')
plt.close()

meanIoU = meanIU.mean(axis=1)

plt.plot(x, meanIoU)
plt.title('Mean IoU by epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean IoU')
plt.savefig('miou.jpg')
plt.close()

