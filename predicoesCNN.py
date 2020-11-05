import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from recDigitoPunho import x_test
from CNN import model

predictions = model.predict_classes(x_test)

plt.figure(figsize=(7,14))
for i in range(0, 8):
    random_num = np.random.randint(0, len(x_test))
    img = x_test[random_num]
    plt.subplot(6,4,i+1)
    plt.margins(x = 20, y = 20)
    plt.title('Predição: ' + str(predictions[random_num]))
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()

submission = pd.DataFrame({'ImageID' : pd.Series(range(1,28001)), 'Label' : predictions})
submission.to_csv("submission.csv",index=False)