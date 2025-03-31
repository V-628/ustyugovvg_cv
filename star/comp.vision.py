# 1
# import numpy as np
# import matplotlib.pyplot as plt

# image = np.zeros((500, 500))
# bs = 25

# for y in range(0, image.shape[0], bs):
#     for x in range(0, image.shape[1], bs):
#         image[y:y+y,x:x+bs] = num
#         num =+ 1

# plt.imshow()
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# image = np.ones((600, 600))
# d = 250
# y = np.arange(600).reshape(600, 1)
# x = np.arange(600).reshape(1, 600)
# mask = x ** 2 + y ** 2 < (d/2)**2


# 2
#  plt.imshow()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# image = np.ones((600, 600))
# d = 250

# y = np.arange(600).reshape(600, 1) - 300
# x = np.arange(600).reshape(1, 600) - 300
# mask = x**2 + y**2 < (d/2)**2

# plt.imshow(mask)
# plt.show()


# 3
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.datasets import face

# def descritize(image, levels=5):
#     mn = image.min()
#     mx = np.max(image)
#     result = image.copy()
#     percent = (result - mn)/(mx - mn)
#     result = (percent * levels).astype("uint16")
    
#     return result

# image = face(gray=True)


# plt.imshow(descritize(image, 2))
# plt.show()

# 4
# import matplotlib.pyplot as plt
# import numpy as np

# def descritize(image, levels=5):
#     result = (percent * levels).astype("uint32")
#     return result
# def block_mean(image, ybc = 10, xbc = 10):
#     result = image.copy()
#     ybs = image.shape[0]//ybs
#     xbs = image.shape[1]//xbs
#     for y range(0, image.shape)
        
# 5 определение зашумлённости
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.datasets import face

# def mse(reference, noised):
#     return ((reference - noised)**2).sum()/reference.size

# def psnr(reference, noised):
#     return 20* np.log10(reference.max() / np.sqrt(noised))

# image = face(gray = True)
# noised = image.copy()
# noise_percent = 0.1
# noised_pixels = int(noise_percent * image.size)
# y_pos, x_pos = np.meshgrid(np.arange(image.shape[0]),
#                             np.arange(image.shape[1]))
                    
# y_pos = y_pos.flatten()
# x_pos = x_pos.flatten()
# plt.figure()
# plt.subplot(121)
# plt.title("Original")
# plt.imshow(image)
# plt.subplot(122)
# plt.title("Noised")
# plt.imshow(noised)
# plt.show()

# # 6 PSNR
# def mse(reference, noised):
#     return ((reference - noised)**2).sum()/reference.size

# def psnr(reference, noised):
#     return 20* np.log10(reference.max() / np.sqrt(noised))

# 7 поиск объектов
# import os
# print("Current Working Directory:", os.getcwd())

# import matplotlib.pyplot as plt
# import numpy as np

# image = np.load(r"c:\Users\Tmbk\Desktop\JOB\Kurs_1_Semac_2\№3_Computer_vision\ex5npy.txt")

# print(image.shape)
# plt.imshow(image)
# plt.show()