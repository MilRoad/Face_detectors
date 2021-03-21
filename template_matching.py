import cv2
from matplotlib import pyplot as plt
import os


class TemplateMatching:
    methods = ['cv2.TM_CCOEFF']

    def matcher(self, template_path, image_path):
        image = cv2.imread(image_path, 0)
        template = cv2.imread(template_path, 0)
        w, h = template.shape[::-1]

        for meth in self.methods:
            img = image.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(img, top_left, bottom_right, 0, 5)

            plt.subplot(121)
            plt.imshow(res, cmap='gray')
            plt.title('Matching Result')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122)
            plt.imshow(img, cmap='gray')
            plt.title('Detected Point')
            plt.xticks([])
            plt.yticks([])
            plt.suptitle(meth)

            plt.show()


if __name__ == '__main__':
    file_directory = 'photo'
    images = [os.path.join(file_directory, f'photo{i}.jpg') for i in range(1, 8)]
    file_directory = 'templates'
    templates = [os.path.join(file_directory, f'photo{i}.jpg') for i in range(1, 4)]
    matching_instance = TemplateMatching()
    for template in templates:
        for image in images:
            print(f'finding {template} in {image}')
            matching_instance.matcher(template, image)