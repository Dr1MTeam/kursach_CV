import cv2
import numpy as np
import random

# Функция для регулировки яркости и контрастности
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    img = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return img

# Функция для наложения дефекта на изображение ноутбука и возврата bbox
def apply_defect(laptop_image, defect_image, mask, radius=20, brightness=0, contrast=1.0):
    h, w = laptop_image.shape[:2]
    
    # Генерация случайной позиции внутри маски
    while True:
        x = random.randint(radius, w - radius)
        y = random.randint(radius, h - radius)
        if mask[y, x] > 0:
            break
    
    # Создание новой маски в форме круга с размытыми краями
    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(new_mask, (x, y), radius, 255, -1)
    # new_mask = cv2.GaussianBlur(new_mask, (51, 51), 20)  # Размытие маски

    # Масштабирование дефекта для улучшенной видимости
    defect_resized = cv2.resize(defect_image, (2 * radius, 2 * radius))

    # Случайный угол поворота от 0 до 360 градусов
    angle = random.uniform(0, 360)
    
    # Центр для поворота
    center = (radius, radius)
    
    # Матрица поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Применение поворота
    defect_rotated = cv2.warpAffine(defect_resized, M, (2 * radius, 2 * radius), flags=cv2.INTER_LINEAR)

    # Создание маски для черных пикселей
    black_mask = np.all(defect_rotated == [0, 0, 0], axis=-1).astype(np.uint8)

    kernel = np.ones((2, 2), np.uint8)  # Ядро 3x3, чтобы расширить на 2 пикселя (делаем 2 итерации)
    black_mask_dilated = cv2.dilate(black_mask, kernel, iterations=2)

    # Вычитание черных областей из общей маски
    mask_region = new_mask[y - radius:y + radius, x - radius:x + radius]
    mask_region[black_mask_dilated == 1] = 0

    

    # Регулировка яркости и контрастности
    defect_rotated = adjust_brightness_contrast(defect_rotated, brightness, contrast)
    
    # Координаты области для вставки дефекта
    x1, y1 = x - radius, y - radius
    x2, y2 = x + radius, y + radius
    
    # Создание альфа-канала для плавного наложения дефекта
    alpha = mask_region / 255.0

    # Наложение дефекта с учетом маски (игнорируем черные области)
    for c in range(0, 3):
        laptop_image[y1:y2, x1:x2, c] = (alpha * defect_rotated[:, :, c] +
                                         (1 - alpha) * laptop_image[y1:y2, x1:x2, c])


     # Расширение маски на 1 пиксель
    expanded_mask = cv2.dilate(new_mask, np.ones((2, 2), np.uint8), iterations=1)

    # Сжатие маски на 2 пикселя
    eroded_mask = cv2.erode(new_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Вычитание сжатой маски из расширенной маски
    blurred_mask = cv2.subtract(expanded_mask, eroded_mask)

    # Размытие laptop_image по краевой маске
    blurred_laptop_image = cv2.GaussianBlur(laptop_image, (21, 21), 10)  # Размытие всего изображения
    laptop_image = np.where(blurred_mask[:, :, None] > 0, blurred_laptop_image, laptop_image)  # Заменяем область размытым изображением


    # Размытие laptop_image по краевой маске
    blurred_laptop_image = cv2.GaussianBlur(laptop_image, (21, 21), 10)  # Размытие всего изображения
    laptop_image = np.where(blurred_mask[:, :, None] > 0, blurred_laptop_image, laptop_image)  # Заменяем область размытым изображением

    # Вычисление bbox в формате YOLO
    bbox_x_center = (x1 + x2) / 2 / w  # нормализованный x центра
    bbox_y_center = (y1 + y2) / 2 / h  # нормализованный y центра
    bbox_width = (x2 - x1) / w         # нормализованная ширина
    bbox_height = (y2 - y1) / h        # нормализованная высота

    return laptop_image, (bbox_x_center, bbox_y_center, bbox_width, bbox_height)

def draw_dead_pixel(laptop_img, mask_resized):
    # Получаем координаты всех белых пикселей (где маска равна 255)
    screen_pixels = np.where(mask_resized == 255)
    
    if len(screen_pixels[0]) == 0 or len(screen_pixels[1]) == 0:
        print("Маска не содержит области для дефекта.")
        return laptop_img, None
    
    # Выбираем случайные координаты внутри маски
    idx = random.randint(0, len(screen_pixels[0]) - 1)
    y, x = screen_pixels[0][idx], screen_pixels[1][idx]

    # Случайный размер пикселя (например, от 2x2 до 10x10)
    pixel_size = random.randint(2, 10)

    # Случайный цвет пикселя
    pixel_color = [random.randint(0, 255) for _ in range(3)]  # RGB
    
    # Рисуем пиксель на изображении
    top_left = (x - pixel_size // 2, y - pixel_size // 2)
    bottom_right = (x + pixel_size // 2, y + pixel_size // 2)
    cv2.rectangle(laptop_img, top_left, bottom_right, pixel_color, -1)

    # Вычисляем bounding box в формате YOLO
    img_h, img_w = laptop_img.shape[:2]
    
    # Центр bounding box
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    
    # Ширина и высота bounding box
    bbox_width = bottom_right[0] - top_left[0]
    bbox_height = bottom_right[1] - top_left[1]

    # YOLO формат: (x_center, y_center, width, height) в относительных координатах (0-1)
    bbox = [
        center_x / img_w,
        center_y / img_h,
        bbox_width / img_w,
        bbox_height / img_h
    ]

    return laptop_img, bbox

def mask_magic(mask, ex = (3,3), er = (3,3)):
    # Расширение маски на 2 пиксель
    expanded_mask = cv2.dilate(mask, np.ones(ex, np.uint8), iterations=1)

    # Сжатие маски на 2 пикселя
    eroded_mask = cv2.erode(mask, np.ones(er, np.uint8), iterations=1)

    return cv2.subtract(expanded_mask, eroded_mask) # Маска краёв

# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠉⠉⠛⠻⢿⣿⠿⠛⠋⠁⠈⠙
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠴⣿⠟⠉⠄⠄⠈⡀⠄⠄⠄⠄⠄⠄
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠄⣾⣿⣿⣿⣿⣿⠿⠿⠿⠛⠛⠉⠉⠄⠄⠄⠄⠄⠄⠄⠉⢁⠄⠄⠈⠄⠄⠄⠄⢀⡇⠄⠄⠄⠄⠄⠄
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠄⠄⣀⣿⠿⠛⠉⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢠⡀⠄⠄⠄⢀⣠⣾⣿⠄⠄⠐⢦⡀⠄
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⣠⡾⠋⠁⠄⠄⠄⠄⠄⠄⠄⠄⣤⣤⣄⣀⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠛⠻⠿⠿⠟⠛⠋⢷⣄⠄⠄⠹⣦
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⡟⠛⠛⠛⠛⠯⠶⣤⣀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠻⣷⣤⡀⠘
# ⣿⣿⣿⣿⣿⣿⣿⡿⠃⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⡇⠄⠄⠄⠄⠄⠄⠄⠉⠑⠢⣀⠈⠢⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠙⣿⣿⣷
# ⣿⣿⣿⣿⣿⣿⠏⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢰⠄⠄⡇⠄⠄⠄⣀⠄⠒⠄⠄⠄⠄⠄⠑⠢⡙⡳⣄⠄⠄⠄⠈⠄⠄⠄⠄⠈⠻⣿
# ⣿⣿⣿⣿⡿⠃⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢸⡆⠄⠃⢀⡴⠚⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠱⠈⠳⡄⠄⠄⠄⢂⠄⠄⠄⠄⠘
# ⣿⣿⣿⡿⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⣾⡀⢐⠉⠄⠄⠄⠄⠄⠄⠄⠄⠄⢀⣀⣀⣴⡀⠁⠄⠙⢦⠄⠄⠈⣧⡀⠄⠄⠄
# ⣿⣿⣿⠃⠄⠄⠄⡇⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⡏⠳⡈⡀⠄⠄⠄⠄⠄⢀⣤⣶⣿⡿⠿⠽⠿⠿⣿⣷⣶⣌⡳⡀⠄⢹⣷⡄⠄⠄
# ⣿⡟⠁⠄⠄⠄⠄⣷⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢸⠄⠑⢥⠄⠄⡾⠋⣰⡿⡟⠊⠄⠚⣿⣿⣿⣶⣄⠄⠉⢹⠄⢳⠄⢸⣿⣿⡄⠄
# ⣿⠇⠄⠄⠄⠄⠄⢹⣇⠄⠄⠄⠂⠄⠄⠄⠄⠄⠄⠄⠘⡄⠄⠈⠄⠈⠄⠰⢻⠋⠄⣀⣀⣠⣿⣿⣿⣿⣿⣇⠄⠈⠄⠄⢃⢘⡏⢿⣿⡄
# ⡿⠄⠄⠄⠄⠄⠄⣿⠈⠣⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢃⠄⠄⠄⠄⠄⠄⠋⠄⠄⢿⣿⣿⣿⣿⣿⡿⠟⠁⠄⠄⠄⠄⠘⣼⡇⠈⢿⣿
# ⡇⠄⠄⠄⠄⡆⠄⣿⠄⠄⡨⠂⠄⡀⠄⠄⠄⠠⣀⠄⠄⠘⡄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠙⠻⠿⠛⠁⠄⠄⠄⠄⠄⠄⠄⠄⣿⠄⠄⠈⣿
# ⡇⠄⠄⠄⠄⠄⠄⢸⠄⣐⠊⠄⠄⠄⢉⠶⠶⢂⠈⠙⠒⠂⠠⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠇⠄⠄⠄⠸
# ⠄⣀⠂⢣⡀⠄⠄⠘⣠⠃⠄⠄⠄⠄⣠⣴⣾⠷⠤⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⡀⡙⠄⠈⢧⠄⠡⡀⢉⠄⠄⠄⠄⣴⣿⡫⣋⣥⣤⣀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⡗⠃⠐⠄⠈⣷⡀⢳⡄⠂⠄⠄⣸⣿⡛⠑⠛⢿⣿⣿⣷⡄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⡇⡀⠂⡀⠄⣸⢱⡈⠇⠐⠄⡠⣿⡟⠁⠄⠄⣸⣿⣿⣿⡟⠄⠄⠄⠄⠈⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⣿⡐⡀⠄⢠⠏⠄⢳⡘⡄⠈⠄⢿⡿⠄⢻⣿⣿⣿⡿⠋⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⣿⣧⠐⢀⡏⠄⠄⠄⢳⡴⡀⠄⢸⣿⡄⠄⠉⠛⠋⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⣶⣶⣶⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⣿⣿⣆⠄⠐⡀⠄⠄⠄⢻⣷⡀⠄⠃⠙⠂⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢿⣿⣿⣿⣄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄
# ⣿⣿⣿⣆⠄⠙⣄⠄⠄⠄⠱⣕⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠻⣿⣿⣿⣦⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⣴
# ⣿⣿⣿⣿⣧⡀⠘⢦⡀⠄⠄⠈⢢⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠘⠿⣿⣿⣇⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⣠⣾⣿
# ⣿⣿⣿⣿⣿⣷⢄⠈⠻⣆⠄⠄⠄⠑⢄⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠛⠿⠄⠄⠄⠄⠄⠄⠄⠄⢀⣴⣾⣿⣿⣿