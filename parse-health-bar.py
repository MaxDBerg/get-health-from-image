from concurrent.futures import ThreadPoolExecutor
import sys
import time
import cv2
import numpy as np
import zmq
import mss

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

data_socket = context.socket(zmq.PUSH)
data_socket.bind("tcp://*:5556")


is_running = False
last_health_data_sent = {
    "player1": 100,
    "player2": 100,
    "player3": 100,
    "player4": 100,
    "player5": 100,
}
player_wait_frames = {
    "player1": 0,
    "player2": 0,
    "player3": 0,
    "player4": 0,
    "player5": 0,
}
player_recovery_wait_frames = {
    "player1": 0,
    "player2": 0,
    "player3": 0,
    "player4": 0,
    "player5": 0,
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


def is_image_red(frame):
    dim = (frame.shape[1], frame.shape[0])

    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([115, 25, 14])
    upper_red = np.array([255, 0, 0])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    grayscale_frame = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(grayscale_frame, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred_frame, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return -2

    return -1


def parse_hp(hp_area):
    width = int(hp_area.shape[1] * 5)
    height = int(hp_area.shape[0] * 5)
    dim = (width, height)

    resized = cv2.resize(hp_area, dim, interpolation=cv2.INTER_AREA)
    grayscale_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    health_data = process_grayscale_image(grayscale_resized)
    if health_data == -1:
        health_data = is_image_red(hp_area)
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
    global last_health_data_sent
    global player_wait_frames
    health_data = {
        "player1": health_bars[0],
        "player2": health_bars[1],
        "player3": health_bars[2],
        "player4": health_bars[3],
        "player5": health_bars[4],
    }
    sending_data = {}

    for player, new_health in health_data.items():
        if (
            last_health_data_sent[player] > 0
            and last_health_data_sent[player] <= 100
            and new_health == -1
        ):
            if player_wait_frames[player] == 0:
                player_wait_frames[player] = 3  # Start the 10-frame countdown
            if player_wait_frames[player] > 0:
                player_wait_frames[player] -= 1
                if player_wait_frames[player] == 1:
                    player_wait_frames[player] -= 2
                continue
            elif player_wait_frames[player] == 0:
                sending_data[player] = -1  # After 10 frames, set health to -1
                last_health_data_sent[player] = -1
                continue

        if last_health_data_sent[player] <= 35 and any(
            percent == new_health for percent in [97, 98, 99, 100]
        ):
            if player_recovery_wait_frames[player] == 0:
                player_recovery_wait_frames[player] = 10
            if player_recovery_wait_frames[player] > 0:
                player_recovery_wait_frames[player] -= 1
                continue
            elif player_recovery_wait_frames[player] == 0:
                sending_data[player] = 100
                last_health_data_sent[player] = 100
                continue
        elif last_health_data_sent[player] <= 50 and any(
            percent == new_health for percent in [97, 98, 99, 100]
        ):
            if player_recovery_wait_frames[player] == 0:
                player_recovery_wait_frames[player] = 2
            if player_recovery_wait_frames[player] > 0:
                player_recovery_wait_frames[player] -= 1
                continue
            elif player_recovery_wait_frames[player] == 0:
                sending_data[player] = 100
                last_health_data_sent[player] = 100
                continue

        # Normal case: if health has changed and is not -2, update the health data immediately
        if last_health_data_sent[player] != new_health and new_health != -2:
            player_wait_frames[player] = 0  # Reset the main wait frames
            player_recovery_wait_frames[player] = 0  # Reset the recovery wait frames
            sending_data[player] = new_health
            last_health_data_sent[player] = new_health

    if sending_data:
        data_socket.send_json(sending_data)
    print(f"Health percentage for player1: {health_data['player1']}%")
    print(f"Health percentage for player2: {health_data['player2']}%")
    print(f"Health percentage for player3: {health_data['player3']}%")
    print(f"Health percentage for player4: {health_data['player4']}%")
    print(f"Health percentage for player5: {health_data['player5']}%")


def handle_null_bar_error(health_bars, null_bar_error, wait_frames_2):
    mutable_health_array = health_bars.copy()
    mutable_health_array.sort()
    if all(val == -1 for val in mutable_health_array[:3]):
        print("\n")
        print(mutable_health_array[:3])
        print("\n")
        if not null_bar_error and wait_frames_2 == 0:
            print("Setting wait_frames...")
            print("Flagged for possible null bar error!")
            return True, 5
        elif wait_frames_2 == 0:
            send_health_data(health_bars)
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
            print("Setting wait_frames...")
            return False, True, 5
        elif wait_frames == 0:
            print("Slowing down process cycle...")
            send_health_data(health_bars)
            return True, False, 0
        elif wait_frames != 0:
            print("Ticking down by one...")
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

                time.sleep(0.01)

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
