import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
import sionna
import json

# 设置GPU和TF配置
gpu_num = 0  # 使用 "" 使用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)  # 设置全局随机种子

# 配置路径参数
base_scene_path = "D:/Sionna/maps/"  # 场景文件基础路径
output_dir = "output_images"  # 输出目录
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# 处理参数设置
start_num = 1  # 起始编号
end_num = 10  # 结束编号
cm_cell_size = (10.0, 10.0)  # 覆盖图单元大小
num_samples = int(20e6)  # 采样次数

# 遍历处理所有场景文件
for num in range(start_num, end_num + 1):
    start_time = time.time()
    scene_dir = f"u{num}"
    scene_file = f"u{num}.xml"
    scene_path = os.path.join(base_scene_path, scene_dir, scene_file)

    # 检查文件是否存在
    if not os.path.exists(scene_path):
        print(f"场景文件 {scene_path} 不存在，跳过处理")
        continue

    try:
        print(f"正在处理场景 {scene_file}...")

        # 加载场景
        scene = load_scene(scene_path)

        # 配置天线阵列
        scene.tx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V"
        )
        scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="cross"
        )

        # 清理现有设备
        if "tx" in scene.transmitters:
            scene.remove("tx")
        if "rx" in scene.receivers:
            scene.remove("rx")

        # 添加新设备
        tx = Transmitter(
            name="tx",
            position=[0, 0, 105],  # 统一发射机位置
            orientation=[0, 0, 0]
        )
        scene.add(tx)

        rx = Receiver(
            name="rx",
            position=[45, 90, 1.5],  # 统一接收机位置
            orientation=[0, 0, 0]
        )
        scene.add(rx)

        # 设置场景参数
        scene.frequency = 2.14e9
        scene.synthetic_array = True

        # 生成覆盖图
        cm = scene.coverage_map(
            max_depth=2,
            diffraction=True,
            cm_cell_size=cm_cell_size,
            combining_vec=None,
            precoding_vec=None,
            num_samples=num_samples
        )

        # 生成并保存图像
        plt.figure()
        cm.show(tx=0)
        output_path = os.path.join(output_dir, f"u{num}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 清理TensorFlow计算图
        tf.keras.backend.clear_session()

        process_time = time.time() - start_time
        print(f"成功处理 {scene_file}，耗时 {process_time:.2f} 秒")

    except Exception as e:
        print(f"处理 {scene_file} 时发生错误: {str(e)}")
        continue

print("所有场景处理完成！")