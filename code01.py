from controller import Robot, Motor, DistanceSensor, Camera, Emitter, GPS, Gyro
import struct
import numpy as np
import math
import random
import time
import cv2

robot = Robot()

#########################################################################################
timeStep = 32
max_velocity = 3.5  # Увеличена скорость с 2 до 3.5
swamp_colour = b'\x12\x1b \xff'
white = b'\xfc\xfc\xfc\xff'
black = b'<<<\xff'

# Параметры для улучшенного распознавания
VICTIM_AREA_THRESHOLD = 400  # Уменьшен порог для обнаружения маленьких жертв
COLOR_THRESHOLD = 100  # Порог для определения цвета (было 44)
TURN_DELAY = 180  # Уменьшена задержка поворота (было 250)
FORWARD_DELAY = 200  # Уменьшена задержка движения вперед
SPIN_DELAY = 600  # Уменьшена задержка разворота (было 1000)

# Цветовые диапазоны для жертв (в формате HSV)
VICTIM_COLORS = {
    'R': ([0, 50, 50], [10, 255, 255]),    # Красный
    'G': ([40, 50, 50], [80, 255, 255]),   # Зеленый
    'B': ([100, 50, 50], [130, 255, 255]), # Синий
    'Y': ([20, 50, 50], [35, 255, 255]),   # Желтый
    'T': ([0, 0, 0], [180, 50, 255])       # Темный (болото)
}

###############################################################################################

def delay(ms):
    """Оптимизированная функция задержки"""
    initTime = robot.getTime()
    targetTime = initTime + ms / 1000.0
    while robot.step(timeStep) != -1 and robot.getTime() < targetTime:
        pass

def set_motor_speed(left_speed, right_speed):
    """Безопасная установка скорости моторов"""
    motor_L.setVelocity(float(left_speed))
    motor_R.setVelocity(float(right_speed))

def Forward(duration=FORWARD_DELAY):
    """Улучшенное движение вперед"""
    set_motor_speed(max_velocity, max_velocity)
    delay(duration)

def stop_motors():
    """Остановка моторов"""
    set_motor_speed(0, 0)

def spin_R(duration=SPIN_DELAY):
    """Поворот направо"""
    set_motor_speed(max_velocity * 1.5, -max_velocity * 1.5)
    delay(duration)

def spin_L(duration=SPIN_DELAY):
    """Поворот налево"""
    set_motor_speed(-max_velocity * 1.5, max_velocity * 1.5)
    delay(duration)

def turn_slight_right(duration=150):
    """Плавный поворот направо"""
    set_motor_speed(max_velocity, max_velocity * 0.5)
    delay(duration)

def turn_slight_left(duration=150):
    """Плавный поворот налево"""
    set_motor_speed(max_velocity * 0.5, max_velocity)
    delay(duration)

def getColor():
    """Получение цвета с датчика"""
    img = colour_camera.getImage()
    return colour_camera.imageGetGray(img, colour_camera.getWidth(), 0, 0)

def detect_victim_advanced(image_data, camera):
    """
    Улучшенное распознавание жертв с определением типа
    Возвращает список найденных жертв с координатами и типом
    """
    victims = []
    
    # Преобразование изображения
    img = np.array(np.frombuffer(image_data, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)))
    
    # Конвертация в HSV для лучшего распознавания цветов
    img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Улучшенная бинаризация
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Морфологические операции для улучшения контуров
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > VICTIM_AREA_THRESHOLD:
            # Определение центра контура
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Определение цвета жертвы
                victim_type = identify_victim_color(img_hsv, cx, cy)
                
                victims.append({
                    'x': cx,
                    'y': cy,
                    'type': victim_type,
                    'area': area
                })
                print(f"Victim detected at x={cx}, y={cy}, type={victim_type}, area={area}")
    
    return victims

def identify_victim_color(img_hsv, center_x, center_y):
    """
    Определение типа жертвы по цвету в центре контура
    """
    h, w = img_hsv.shape[:2]
    radius = 15
    
    # Область вокруг центра
    x1 = max(0, center_x - radius)
    x2 = min(w, center_x + radius)
    y1 = max(0, center_y - radius)
    y2 = min(h, center_y + radius)
    
    roi = img_hsv[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 'U'  # Unknown
    
    # Определение доминирующего цвета
    for vtype, (lower, upper) in VICTIM_COLORS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(roi, lower, upper)
        if np.sum(mask) > 100:  # Если найдено достаточно пикселей
            return vtype
    
    return 'U'

def report(victimType):
    """Отправка отчета о жертве"""
    stop_motors()
    delay(1000)  # Уменьшена задержка перед отчетом
    
    if isinstance(victimType, str):
        victimType = bytes(victimType, "utf-8")
    
    pos = Gps.getValues()
    posX = int(pos[0] * 100)
    posZ = int(pos[2] * 100)
    
    message = struct.pack("i i c", posX, posZ, victimType)
    emitter.send(message)
    print(f"Report sent: Type={victimType.decode() if isinstance(victimType, bytes) else victimType}, Pos=({posX}, {posZ})")
    robot.step(timeStep)

def left_wall_moving():
    """Улучшенный алгоритм движения вдоль стены"""
    gps_history_x = []
    
    while robot.step(timestep) != -1:
        # Получение данных с камер
        img1 = camera1R.getImage()
        img2 = camera2L.getImage()
        img3 = camera3F.getImage()
        
        # Распознавание жертв
        victims = []
        victims.extend(detect_victim_advanced(img1, camera1R))
        victims.extend(detect_victim_advanced(img2, camera2L))
        victims.extend(detect_victim_advanced(img3, camera3F))
        
        # Отправка отчетов о найденных жертвах
        for victim in victims:
            report(victim['type'])
        
        # Получение данных с датчиков
        left_wall_dist = distance_sens[0].getValue()
        front_left_dist = distance_sens[1].getValue()
        front_right_dist = distance_sens[2].getValue()
        right_wall_dist = distance_sens[3].getValue()
        left_corner_dist = distance_sens[5].getValue()
        front_dist = distance_sens[6].getValue()
        
        # GPS трекинг
        pos = Gps.getValues()
        x_coord = int(pos[0] * 10)
        
        if x_coord not in gps_history_x:
            gps_history_x.append(x_coord)
            report("T")  # Отчет о новом участке
        
        # Проверка болота
        if getColor() > COLOR_THRESHOLD:
            spin_R()
            continue
        
        # Логика движения
        left_speed = max_velocity
        right_speed = max_velocity
        
        # Приоритет: стена спереди
        if front_dist < 0.15 or front_left_dist < 0.1 or front_right_dist < 0.1:
            # Поворот вправо
            set_motor_speed(max_velocity, -max_velocity)
            delay(TURN_DELAY)
            continue
        
        # Движение вдоль левой стены
        if left_wall_dist < 0.12:
            # Слишком близко к левой стене - отворачиваем
            right_speed = max_velocity * 0.7
            left_speed = max_velocity
        elif left_wall_dist > 0.25:
            # Слишком далеко от левой стены - поворачиваем к ней
            left_speed = max_velocity * 0.7
            right_speed = max_velocity
        else:
            # Оптимальная дистанция - едем прямо
            left_speed = max_velocity
            right_speed = max_velocity
        
        # Коррекция при приближении к углу
        if left_corner_dist < 0.12:
            left_speed = max_velocity
            right_speed = max_velocity * 0.6
        
        set_motor_speed(left_speed, right_speed)
        delay(50)  # Короткая задержка для стабильности

#########################################################################################

# Инициализация устройств
timestep = int(robot.getBasicTimeStep())

# Моторы
motor_L = robot.getDevice("wheel2 motor")
motor_R = robot.getDevice("wheel1 motor")
motor_L.setPosition(float('inf'))
motor_R.setPosition(float('inf'))
motor_L.setVelocity(0.0)
motor_R.setVelocity(0.0)

# Датчики расстояния
distance_sens = []
for i in range(1, 8):
    distance_sens.append(robot.getDevice(f"distance sensor{i}"))
for i in range(7):
    distance_sens[i].enable(timestep)

# Камеры
colour_camera = robot.getCamera("colour_sensor")
colour_camera.enable(timestep)

camera1R = robot.getDevice("camera1")
camera2L = robot.getDevice("camera2")
camera3F = robot.getDevice("camera3")
camera1R.enable(timestep)
camera2L.enable(timestep)
camera3F.enable(timestep)

# GPS и коммуникация
Gps = robot.getDevice("gps")
Gps.enable(timestep)

emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

print("Robot initialized. Starting navigation...")

#########################################################################################

# Основной цикл
GPS_his_X = []

while robot.step(timestep) != -1:
    # Начальное определение направления движения
    if distance_sens[0].getValue() > distance_sens[3].getValue() or \
       abs(distance_sens[0].getValue() - distance_sens[3].getValue()) < 0.05:
        # Поворот для выравнивания
        set_motor_speed(max_velocity, -max_velocity)
        delay(500)
    else:
        GPS_his_X = []
        left_wall_moving()
