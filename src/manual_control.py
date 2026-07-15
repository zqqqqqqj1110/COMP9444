import msvcrt
from pathlib import Path

import airsim


MOVE_SPEED = 2.0       # 水平速度，单位 m/s
VERTICAL_SPEED = 1.0   # 垂直速度，单位 m/s
MOVE_DURATION = 0.5    # 每次按键执行时间，单位秒
YAW_RATE = 30.0        # 旋转速度，单位 degree/s
YAW_DURATION = 0.5

SAVE_FILE = Path("results/selected_targets.txt")


def get_position(client: airsim.MultirotorClient):
    state = client.getMultirotorState()
    return state.kinematics_estimated.position


def print_position(client: airsim.MultirotorClient, prefix: str = "当前位置"):
    position = get_position(client)

    x = position.x_val
    y = position.y_val
    z = position.z_val

    print(f"{prefix}: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    return x, y, z


def move_body(
    client: airsim.MultirotorClient,
    vx: float,
    vy: float,
    vz: float,
):
    """
    Body frame:
    x 正方向：无人机当前朝向的前方
    y 正方向：无人机当前朝向的右方
    z 正方向：向下
    """
    client.moveByVelocityBodyFrameAsync(
        vx,
        vy,
        vz,
        MOVE_DURATION,
    ).join()

    client.hoverAsync().join()
    print_position(client)


def save_target(client: airsim.MultirotorClient):
    x, y, z = print_position(client, "已选择目标点")

    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with SAVE_FILE.open("a", encoding="utf-8") as file:
        file.write(f"{x:.6f},{y:.6f},{z:.6f}\n")

    print(f"坐标已保存到：{SAVE_FILE.resolve()}")


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True)
    client.armDisarm(True)

    position = get_position(client)

    # 如果无人机尚未起飞，则自动起飞并移动到约 3 米高度
    if position.z_val > -1.0:
        print("无人机正在起飞……")
        client.takeoffAsync().join()
        client.moveToZAsync(-3.0, 1.0).join()
        client.hoverAsync().join()

    print_position(client, "起始位置")

    print(
        """
键盘控制：

W：向前
S：向后
A：向左
D：向右

R：上升
F：下降

Q：向左旋转
E：向右旋转

P：打印并保存当前位置
H：悬停
L：降落并退出
Esc：悬停并退出

注意：请保持当前 PowerShell 窗口获得键盘焦点。
"""
    )

    try:
        while True:
            key = msvcrt.getwch().lower()

            if key == "w":
                move_body(client, MOVE_SPEED, 0.0, 0.0)

            elif key == "s":
                move_body(client, -MOVE_SPEED, 0.0, 0.0)

            elif key == "a":
                move_body(client, 0.0, -MOVE_SPEED, 0.0)

            elif key == "d":
                move_body(client, 0.0, MOVE_SPEED, 0.0)

            elif key == "r":
                # NED 中负 z 表示上升
                move_body(client, 0.0, 0.0, -VERTICAL_SPEED)

            elif key == "f":
                # NED 中正 z 表示下降
                move_body(client, 0.0, 0.0, VERTICAL_SPEED)

            elif key == "q":
                client.rotateByYawRateAsync(
                    -YAW_RATE,
                    YAW_DURATION,
                ).join()
                client.hoverAsync().join()
                print_position(client)

            elif key == "e":
                client.rotateByYawRateAsync(
                    YAW_RATE,
                    YAW_DURATION,
                ).join()
                client.hoverAsync().join()
                print_position(client)

            elif key == "p":
                save_target(client)

            elif key == "h":
                client.hoverAsync().join()
                print_position(client, "悬停位置")

            elif key == "l":
                print("正在降落……")
                client.landAsync().join()
                client.armDisarm(False)
                client.enableApiControl(False)
                break

            elif key == "\x1b":  # Esc
                client.hoverAsync().join()
                print_position(client, "退出位置")
                break

    except KeyboardInterrupt:
        client.hoverAsync().join()
        print("\n已停止控制。")


if __name__ == "__main__":
    main()