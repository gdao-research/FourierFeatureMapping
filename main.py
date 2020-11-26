import matplotlib.pyplot as plt
import numpy as np
from rff.trainer import Trainer

# test a trainable parameters instead of random gaussian
myTrainer1 = Trainer(preset=False)
psnrs, psnrs_test, rgbs = myTrainer1.train(2000, True)
# use raw normalized indices without transformation
myTrainer2 = Trainer()
psnrs_woB, psnrs_test_woB, rgbs_woB = myTrainer2.train(2000, False)
# random gaussian from paper
myTrainer3 = Trainer(preset=True)
psnr_preset, psnr_test_preset, rgbs_preset = myTrainer3.train(2000, True)


plt.plot(psnrs, label='learnable')
plt.plot(psnrs_test, label='test learnable')
plt.plot(psnrs_woB, label='without B')
plt.plot(psnrs_test_woB, label='test without B')
plt.plot(psnr_preset, label='random gaussian')
plt.plot(psnr_test_preset, label='test random gaussian')
plt.legend()
plt.savefig('images/curve.png')
plt.close()

fig = plt.figure(figsize=(16, 4))
fig.add_subplot(141)
plt.imshow(rgbs_woB)
plt.title('none')
fig.add_subplot(142)
plt.imshow(rgbs_preset)
plt.title('random gaussian')
fig.add_subplot(143)
plt.imshow(rgbs)
plt.title('learnable')
fig.add_subplot(144)
plt.imshow(myTrainer1.test_data[1])
plt.title('ground truth')
plt.savefig('images/output.png')