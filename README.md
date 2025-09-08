# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program and Output:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load face image
faceImage = cv2.imread('photo1.jpeg')
plt.imshow(faceImage[:,:,::-1]); plt.title("Face")
print("Face shape:", faceImage.shape)
```

<img width="409" height="516" alt="image" src="https://github.com/user-attachments/assets/ff2f12cf-5733-4e04-aad4-99f02fabad9b" />

```python
glassJPG = cv2.imread('glass.jpg')
plt.imshow(glassJPG[:,:,::-1]); plt.title("glassJPG")
print("Glass shape:", glassJPG.shape)
```

<img width="657" height="414" alt="image" src="https://github.com/user-attachments/assets/d91df880-68b1-48de-bd55-781bb02a516c" />

```python
glassBGR = glassJPG[:,:,0:3]
glassGray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)
_, glassMask1 = cv2.threshold(glassGray, 240, 255, cv2.THRESH_BINARY_INV)  # detect non-white

plt.figure(figsize=[15,15])
#Show sunglasses color channels
plt.subplot(121)
plt.imshow(glassBGR[:,:,::-1])  # BGR → RGB
plt.title('Sunglass Color channels')
```

<img width="701" height="454" alt="image" src="https://github.com/user-attachments/assets/19cb833a-17e3-4db6-a4d7-29608645f35a" />

```python
#Show generated mask
plt.subplot(122)
plt.imshow(glassMask1, cmap='gray')
plt.title('Sunglass Mask (generated)')
```
<img width="378" height="255" alt="image" src="https://github.com/user-attachments/assets/cc42cd51-9757-45de-b9f9-7c5793ad82b1" />

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load images
faceImage = cv2.imread('photo1.jpeg')
glassJPG = cv2.imread('glass.jpg')

#Check if images loaded correctly
if faceImage is None or glassJPG is None:
    print("Error: Check your file paths!")
else:
    face_h, face_w, _ = faceImage.shape

    #Resize glasses to ~50% of face width
    new_w = int(face_w * 0.5)
    new_h = int(new_w * glassJPG.shape[0] / glassJPG.shape[1])
    glass_resized = cv2.resize(glassJPG, (new_w, new_h))

    #Create mask
    glass_gray = cv2.cvtColor(glass_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(glass_gray, 240, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    # Adjusted position to place glasses on eyes
    x = int(face_w * 0.25)   # x offset (centered)
    y = int(face_h * 0.25)   # y offset (move up from nose to eyes)

    #ROI on face
    roi = faceImage[y:y+new_h, x:x+new_w]

    if roi.shape[0] > 0 and roi.shape[1] > 0:
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(glass_resized, glass_resized, mask=mask)
        combined = cv2.add(bg, fg)
        faceImage[y:y+new_h, x:x+new_w] = combined

    #Show result
    plt.figure(figsize=[10,10])
    plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
    plt.title("Face with Sunglasses")
    plt.axis("off")
    plt.show()
```

<img width="479" height="623" alt="image" src="https://github.com/user-attachments/assets/dd73bce9-0508-42b6-adde-6afc37327ce3" />

