from concurrent.futures import ThreadPoolExecutor
import sys
import time
import cv2
import numpy as np
import zmq
import mss
import math
from scipy.signal import savgol_filter

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

data_socket = context.socket(zmq.PUSH)
data_socket.bind("tcp://*:5556")


is_running = False
data_filter_tag = ""
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
last_data_bundle = {
    "player1": 100,
    "player2": 100,
    "player3": 100,
    "player4": 100,
    "player5": 100,
}

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
            return -1
    return -1


def process_grayscale_image(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return calculate_health_percentage(frame, contours)

    return -1


def parse_hp(hp_area):
    width = int(hp_area.shape[1] * 5)
    height = int(hp_area.shape[0] * 5)
    dim = (width, height)

    resized = cv2.resize(hp_area, dim, interpolation=cv2.INTER_AREA)
    grayscale_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    health_data = process_grayscale_image(grayscale_resized)
    return health_data


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
    bar_margin = 54
    x = 0

    bars = []
    for i in range(num_players):
        top = i * bar_margin
        bar = frame[top : top + bar_height, x : x + bar_width].copy()
        bars.append(bar)

    result = process_bars(bars)

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


def send_health_data(health_bars):
    global health_data_buffer
    global data_filter_tag
    global player_dead
    global last_data_bundle
    global player_dead_wait_frames
    global player_recovery_wait_frames
    new_health_data = {
        "player1": health_bars[0],
        "player2": health_bars[1],
        "player3": health_bars[2],
        "player4": health_bars[3],
        "player5": health_bars[4],
    }

    for player, new_health in new_health_data.items():
        if new_health == -1:
            new_health = 0
        if last_data_bundle[player] == -1 and player_dead[player]:
            if data_filter_tag != "recovery_flicker_error":
                data_filter_tag = "recovery_flicker_error"
                player_recovery_wait_frames[player] = 0
            if player_recovery_wait_frames[player] == 0:
                player_recovery_wait_frames[player] = 10
            else:
                player_recovery_wait_frames[player] -= 1
                if player_recovery_wait_frames[player] == 0:
                    player_dead[player] = False
                continue
        if len(health_data_buffer[player]) != 0:
            if health_data_buffer[player][
                len(health_data_buffer[player]) - 1
            ] <= 50 and [
                95,
                96,
                97,
                98,
                99,
            ].__contains__(new_health):
                health_data_buffer[player].append(
                    health_data_buffer[player][len(health_data_buffer[player]) - 1]
                )
                continue
            else:
                health_data_buffer[player].append(new_health)
        elif [
            95,
            96,
            97,
            98,
            99,
        ].__contains__(new_health) and last_data_bundle[player] <= 50:
            health_data_buffer[player].append(last_data_bundle[player])
        else:
            health_data_buffer[player].append(new_health)

    if len(health_data_buffer["player1"]) >= 3:
        filtered_health_data = []
        filtered_health_data.append(
            math.ceil(sum(savgol_filter(health_data_buffer["player1"], 3, 1)) / 3)
        )
        filtered_health_data.append(
            math.ceil(sum(savgol_filter(health_data_buffer["player2"], 3, 1)) / 3)
        )
        filtered_health_data.append(
            math.ceil(sum(savgol_filter(health_data_buffer["player3"], 3, 1)) / 3)
        )
        filtered_health_data.append(
            math.ceil(sum(savgol_filter(health_data_buffer["player4"], 3, 1)) / 3)
        )
        filtered_health_data.append(
            math.ceil(sum(savgol_filter(health_data_buffer["player5"], 3, 1)) / 3)
        )

        print("\n")
        print(health_data_buffer)
        print("\n")
        print(savgol_filter(health_data_buffer["player5"], 3, 1))
        print("\n")
        print(filtered_health_data)
        print("\n")

        if filtered_health_data[0] == 0:
            player_dead["player1"] = True

        if filtered_health_data[1] == 0:
            player_dead["player2"] = True

        if filtered_health_data[2] == 0:
            player_dead["player3"] = True

        if filtered_health_data[3] == 0:
            player_dead["player4"] = True

        if filtered_health_data[4] == 0:
            player_dead["player5"] = True

        data_socket.send(bytearray(filtered_health_data))
        last_data_bundle = {
            "player1": filtered_health_data[0],
            "player2": filtered_health_data[1],
            "player3": filtered_health_data[2],
            "player4": filtered_health_data[3],
            "player5": filtered_health_data[4],
        }
        filtered_health_data = []
        health_data_buffer = {
            "player1": [],
            "player2": [],
            "player3": [],
            "player4": [],
            "player5": [],
        }


def handle_null_bar_error(health_bars, null_bar_error, wait_frames_2):
    mutable_health_array = health_bars.copy()
    mutable_health_array.sort()
    if all(val == -1 for val in mutable_health_array[:3]):
        if not null_bar_error and wait_frames_2 == 0:
            print("Setting wait_frames...")
            print("Flagged for possible null bar error!")
            return True, 5
        elif wait_frames_2 == 0:
            send_health_data(health_bars)
            return False, 0
        elif wait_frames_2 != 0:
            return True, wait_frames_2 - 1
    return False, 0


def handle_cv_error(health_bars, standby_process, cv_error, wait_frames):
    if all(percent == 100 for percent in health_bars):
        if standby_process:
            print("Process in standby mode...")
            time.sleep(1)
            return True, False, 0
        elif not cv_error and wait_frames == 0:
            print("Flagged for possible cv error!")
            return False, True, 5
        elif wait_frames == 0:
            print("Slowing down process cycle...")
            return True, False, 0
        elif wait_frames != 0:
            return standby_process, cv_error, wait_frames - 1
    else:
        return False, False, 0


def run_processing_loop():
    global is_running
    wait_frames = 0
    wait_frames_2 = 0
    standby_process = False
    cv_error = False
    null_bar_error = False
    party_region = {
        "top": 268,
        "left": 25,
        "width": 220,
        "height": 300,
    }
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
                frame = capture_screen(region=party_region)
                # target_frame = capture_screen(region=healer_target_region)
                # cv2.imshow("showing frame", target_frame)
                # cv2.waitKey(0)
                health_bar_percentages = process_frame(frame)
                null_bar_error, wait_frames_2 = handle_null_bar_error(
                    health_bar_percentages, null_bar_error, wait_frames_2
                )
                standby_process, cv_error, wait_frames = handle_cv_error(
                    health_bar_percentages, standby_process, cv_error, wait_frames
                )

                if not null_bar_error and not cv_error and not standby_process:
                    send_health_data(health_bar_percentages)

                time.sleep(0.001)

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
