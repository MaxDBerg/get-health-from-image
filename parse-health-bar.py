from concurrent.futures import ThreadPoolExecutor
import os
import time
import cv2
import numpy as np
import zmq
import mss

context = zmq.Context()
socket = context.socket(zmq.REP)
print("Binding to port 5555")
socket.bind("tcp://*:5555")
print("Bound")

is_running = False
retry_triggered = False

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)


def extract_blue_channel(image):
    blue_channel, _, _ = cv2.split(image)
    return blue_channel


def calculate_health_percentage(resized, contours):
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) > 25:
        x, y, w, h = cv2.boundingRect(cnt)
        resized_width = int(resized.shape[1])
        hp_width = w
        if x < 50:
            health_percentage = int((hp_width) * 100 / resized_width)
            print(f"Detected health: {health_percentage}%")
            return health_percentage
        else:
            print(f"Detected health: {-1}%")
            return -1
    return -1


def process_grayscale_image(grayscale_image):
    blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return calculate_health_percentage(grayscale_image, contours)

    print("No health bar detected.")
    return -1


def parse_hp(hp_area):
    width = int(hp_area.shape[1] * 5)
    height = int(hp_area.shape[0] * 5)
    dim = (width, height)

    resized = cv2.resize(hp_area, dim, interpolation=cv2.INTER_AREA)
    # resized = extract_blue_channel(resized)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return process_grayscale_image(resized)


def process_bars(bars):
    health_percentages = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(parse_hp, bar) for bar in bars]

        for future in futures:
            health_percentages.append(future.result())

    return health_percentages


def process_frame(frame):
    # Constants
    num_players = 5
    bar_height = 10
    bar_width = 191
    skull_width = 30
    bar_margin = 54
    x = 0
    skull_x = 200

    bars = []
    for i in range(num_players):
        top = i * bar_margin
        bar = frame[top : top + bar_height, x : x + bar_width].copy()
        # bar = frame[top : top + bar_height, skull_x : skull_x + skull_width].copy()
        bars.append(bar)

    result = process_bars(bars)
    for index, percent in enumerate(result):
        print(f"Health percentage for player {index}: {percent}%")

    return result


def handle_messages():
    global is_running
    try:
        socks = dict(poller.poll(100))  # 100ms timeout
        if socket in socks:
            # Check for a message from Electron
            print("Waiting for a request...")
            message = socket.recv().decode("utf-8")
            print(message)
            print("Sending reply")
            socket.send_string("done")

            if message == "start" and not is_running:
                print("Game focused. Starting capture.")
                is_running = True
            elif message == "stop" and is_running:
                print("Game unfocused. Stopping capture.")
                is_running = False
    except zmq.ZMQError as e:
        print(f"ZMQError occurred {e}")


def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        img = np.array(sct.grab(monitor))
        return img[:, :, :3]


def run_processing_loop():
    global is_running
    global retry_triggered
    while True:
        handle_messages()
        if is_running:
            region = {
                "top": 268,
                "left": 25,
                "width": 220,
                "height": 300,
            }
            frame = capture_screen(region=region)
            # cv2.imshow("showing frame", frame)
            # cv2.waitKey(0)
            health_bar_percentages = process_frame(frame)
            if (
                health_bar_percentages[0] != -1
                and health_bar_percentages[len(health_bar_percentages) - 1] != -1
            ):
                retry_triggered = False
                time.sleep(0.01)
            else:
                retry_triggered = True


if __name__ == "__main__":
    run_processing_loop()
