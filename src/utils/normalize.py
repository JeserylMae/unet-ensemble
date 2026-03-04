import cv2
import numpy as np


class Normalize:
    def __init__(self):
        pass

    def rotate_image_by_face_area(self, img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

        best_rotation = None
        max_face_area = 0

        for rot in rotations:
            rotated = img if rot is None else cv2.rotate(img, rot)
            gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                area = w * h
                if area > max_face_area:
                    max_face_area = area
                    best_rotation = rot

        if best_rotation is None:
            return img

        return img if best_rotation is None else cv2.rotate(img, best_rotation)
    

    '''
    Get image edges. Working for Template B & C.
    '''
    def get_edges(self, img_path):
        loaded_img = cv2.imread(img_path)
        scale = 800 / max(loaded_img.shape)

        img = cv2.resize(loaded_img, None, fx=scale, fy=scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.rotate_image_by_face_area(img)

        blur = cv2.GaussianBlur(img, (11, 11), 100)
        gray = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

        kernel = np.ones((5, 5), np.uint16)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(morphed, 1, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((11, 11)))

        return edges, img
    
    '''
    Get image edges. Working for Template A.
    '''
    def get_edges_template_a(self, img_path):
        loaded_img = cv2.imread(img_path)
        scale = 800 / max(loaded_img.shape)

        img = cv2.resize(loaded_img, None, fx=scale, fy=scale)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.rotate_image_by_face_area(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)

        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(morphed, 30, 100)
        edges = cv2.dilate(edges, None, iterations=1)

        return edges, img
    
    '''
    Get contour ratio
    '''
    def get_aspect_ratio(self, cnt):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h

        return aspect_ratio
    
    '''
    Getting the coordinates of the edges.
    '''
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # top-left
        rect[2] = pts[np.argmax(s)] # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # top-right
        rect[3] = pts[np.argmax(diff)] # bottom-left

        return rect

    '''
    Get max width & height.
    '''
    def get_max_dimensions(self, box):
        pts = self.order_points(box)
        (tl, tr, br, bl) = pts.astype(int)

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        return maxWidth, maxHeight, pts
    
    '''
    Cropping the image.
    '''
    def crop_image(self, edges, orig_img):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        largest_cnt = contours[0]
        cnt_ratio = self.get_aspect_ratio(largest_cnt)

        if 1 < cnt_ratio < 2:
            rect = cv2.minAreaRect(largest_cnt)
            box = cv2.boxPoints(rect)
            # box = box.astype("float32")

            maxWidth, maxHeight, pts = self.get_max_dimensions(box)
        else:
            maxHeight, maxWidth = orig_img.shape[:2]
            pts = np.array([
                [0, 0],
                [maxWidth, 0],
                [maxWidth, maxHeight],
                [0, maxHeight]
            ], dtype="float32")

        scale = 256 / max(maxWidth, maxHeight)
        new_w = int(maxWidth * scale)
        new_h = int(maxHeight * scale)

        x_margin = (256 - new_w) // 2
        y_margin = (256 - new_h) // 2

        dst = np.array([
            [x_margin, 0],                 # Top-Left
            [x_margin + new_w, 0],         # Top-Right
            [x_margin + new_w, new_h], # Bottom-Right
            [x_margin, new_h]          # Bottom-Left
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst)
        warped_img = cv2.warpPerspective(orig_img, M, (256, new_h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

        return warped_img, new_w, new_h
    
    '''
    Add margin and normalize to 256x256.
    '''
    def normalize(self, warped_img, new_w, new_h, resize=False):
        if resize:
            resized_img = cv2.resize(warped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        top = (256 - new_h) // 2
        bottom = 256 - new_h - top
        left = (256 - new_w) // 2
        right = 256 - new_w - left

        # 4. Add the black margins
        final_img = cv2.copyMakeBorder(
            warped_img,
            top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        return final_img
    