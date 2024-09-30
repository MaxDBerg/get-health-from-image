import base64
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
import cv2
import numpy as np
import zmq
import mss
import math
import pytesseract
from scipy.signal import savgol_filter

pytesseract.pytesseract.tesseract_cmd = "E:\\Tesseract\\tesseract.exe"

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

data_socket = context.socket(zmq.PUSH)
data_socket.bind("tcp://*:5556")

executor = ThreadPoolExecutor(max_workers=5)
is_running = False
data_filter_tag = ""
player_regions = {
    "1440": [
        {
            "health": {"top": 29, "left": 0, "width": 191, "height": 7},
        },
        {
            "health": {"top": 82, "left": 0, "width": 191, "height": 7},
        },
        {
            "health": {"top": 135, "left": 0, "width": 191, "height": 7},
        },
        {
            "health": {"top": 189, "left": 0, "width": 191, "height": 7},
        },
        {
            "health": {"top": 242, "left": 0, "width": 191, "height": 7},
        },
    ],
    "1080": [
        {
            "health": {"top": 0, "left": 0, "width": 143, "height": 5},
        },
        {
            "health": {"top": 40, "left": 0, "width": 143, "height": 5},
        },
        {
            "health": {"top": 80, "left": 0, "width": 143, "height": 5},
        },
        {
            "health": {"top": 120, "left": 0, "width": 143, "height": 5},
        },
        {
            "health": {"top": 160, "left": 0, "width": 143, "height": 5},
        },
    ],
}
party_region = {
    "1440": {
        "top": 240,
        "left": 25,
        "width": 220,
        "height": 300,
    },
    "1080": {
        "top": 202,
        "left": 19,
        "width": 160,
        "height": 170,
    },
}
last_data_bundle = {
    "player1": 100,
    "player2": 100,
    "player3": 100,
    "player4": 100,
    "player5": 100,
}
health_data_buffer = {
    "player1": [],
    "player2": [],
    "player3": [],
    "player4": [],
    "player5": [],
}
player_recovery_wait_frames = {
    "player1": 0,
    "player2": 0,
    "player3": 0,
    "player4": 0,
    "player5": 0,
}
player_dead_wait_frames = {
    "player1": 0,
    "player2": 0,
    "player3": 0,
    "player4": 0,
    "player5": 0,
}
player_dead = {
    "player1": False,
    "player2": False,
    "player3": False,
    "player4": False,
    "player5": False,
}
players_data = {}

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)


def calculate_health_percentage(resized, contours):
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) > 25:
        x, y, w, h = cv2.boundingRect(cnt)
        resized_width = int(resized.shape[1])
        hp_width = w
        if x < 5:
            health_percentage = int((hp_width) * 100 / resized_width)
            return health_percentage
        else:
            return 0
    return -1


def process_grayscale_image(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return calculate_health_percentage(frame, contours)

    return 0


def parse_hp(hp_area):
    width = int(hp_area.shape[1] * 5)
    height = int(hp_area.shape[0] * 5)
    dim = (width, height)

    resized = cv2.resize(hp_area, dim, interpolation=cv2.INTER_AREA)
    grayscale_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    health_data = process_grayscale_image(grayscale_resized)
    return health_data


def filter_health_data(health_data_buffer):
    filtered_health_data = []
    for player in ["player1", "player2", "player3", "player4", "player5"]:
        if len(health_data_buffer[player]) >= 3:
            filtered_value = math.ceil(
                min(savgol_filter(health_data_buffer[player], 3, 1))
            )
            filtered_health_data.append(
                max(filtered_value, 0)
            )  # Ensure it's non-negative
        else:
            filtered_health_data.append(0)  # Default value if not enough data
    return filtered_health_data


def send_health_data(players_health):
    global health_data_buffer
    global data_filter_tag
    global player_dead
    global last_data_bundle
    global player_dead_wait_frames
    global player_recovery_wait_frames
    global players_data
    new_health_data = {
        "player1": players_health[0],
        "player2": players_health[1],
        "player3": players_health[2],
        "player4": players_health[3],
        "player5": players_health[4],
    }

    for player, new_health in new_health_data.items():
        if new_health == -1:
            print("Error when parsing health!!!!!")
            health_data_buffer[player].append(last_data_bundle[player])
            continue
        if last_data_bundle[player] == 0 and player_dead[player]:
            if data_filter_tag != "recovery_flicker_error":
                data_filter_tag = "recovery_flicker_error"
                player_recovery_wait_frames[player] = 0
            if player_recovery_wait_frames[player] == 0:
                player_recovery_wait_frames[player] = 10
            player_recovery_wait_frames[player] -= 1
            if player_recovery_wait_frames[player] == 0:
                player_dead[player] = False
            health_data_buffer[player].append(last_data_bundle[player])
            continue
        if len(health_data_buffer[player]) != 0:
            if health_data_buffer[player][
                len(health_data_buffer[player]) - 1
            ] <= 50 and [95, 96, 97, 98, 99].__contains__(new_health):
                health_data_buffer[player].append(
                    health_data_buffer[player][len(health_data_buffer[player]) - 1]
                )
                continue
            else:
                health_data_buffer[player].append(new_health)
        elif [95, 96, 97, 98, 99].__contains__(new_health) and last_data_bundle[
            player
        ] <= 50:
            health_data_buffer[player].append(last_data_bundle[player])
        else:
            health_data_buffer[player].append(new_health)

    if len(health_data_buffer["player1"]) >= 3:
        filtered_health_data = filter_health_data(health_data_buffer)

        print("\n")
        print(health_data_buffer)
        print("\n")
        print(filtered_health_data)
        print("\n")

        for i, player in enumerate(
            ["player1", "player2", "player3", "player4", "player5"]
        ):
            players_data[player]["health"] = filtered_health_data[i]

        last_data_bundle = dict(zip(new_health_data.keys(), filtered_health_data))
        last_data_bundle = {
            "player1": filtered_health_data[0],
            "player2": filtered_health_data[1],
            "player3": filtered_health_data[2],
            "player4": filtered_health_data[3],
            "player5": filtered_health_data[4],
        }
        for player in health_data_buffer.keys():
            health_data_buffer[player] = []


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


def capture_and_process():
    global players_data
    players_data = {}
    players_health = []

    with mss.mss() as sct:
        monitor = party_region[f"{sct.monitors[1]["height"]}"]
        screenshot = np.array(sct.grab(monitor))[:, :, :3]

    for i, regions in enumerate(
        player_regions[f"{sct.monitors[1]["height"]}"],
        start=1,
    ):
        health_roi = regions["health"]

        health_bar_image = screenshot[
            health_roi["top"] : health_roi["top"] + health_roi["height"],
            health_roi["left"] : health_roi["left"] + health_roi["width"],
        ]

        # cv2.imshow("showing frame", health_bar_image)
        # cv2.waitKey(0)

        health_percent = parse_hp(health_bar_image)

        smooth_health_bar_image = cv2.bilateralFilter(health_bar_image, 9, 75, 75)
        _, buffer = cv2.imencode(".png", smooth_health_bar_image[0:1, :, :])
        base64_image = base64.b64encode(buffer).decode("utf-8")

        players_data[f"player{i}"] = {
            "health": health_percent,
            "image": base64_image,
        }

    for i, player in enumerate(["player1", "player2", "player3", "player4", "player5"]):
        players_health.append(players_data[player]["health"])

    send_health_data(players_health)

    json_data = json.dumps(players_data)
    data_socket.send_string(json_data)


def run_processing_loop():
    global is_running
    # healer_target_region = {
    #     "top": 850,
    #     "left": 1932,
    #     "width": 264,
    #     "height": 80,
    # }

    while True:
        try:
            handle_messages()
            if is_running:
                start_time = time.time()
                # target_frame = capture_screen(region=healer_target_region)
                capture_and_process()
                # cv2.imshow("showing frame", health_bars[0])
                # cv2.waitKey(0)

                end_time = time.time()
                print(f"Slow as shit : {(end_time - start_time)} seconds")
            else:
                print("Waiting for start signal...")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            socket.close()
            data_socket.close()
            context.term()
            sys.exit(0)


if __name__ == "__main__":
    run_processing_loop()
