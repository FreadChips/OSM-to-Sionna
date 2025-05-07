import os

gpu_num = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
import imageio.v3 as imageio
import gc
from skimage.measure import block_reduce


if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)
# 路径配置
base_scene_path = "/home/lyus/fzq/mapss"
output_dirs = {
    "default": "default_coverage",  # 原始覆盖图
    "building": "building_coverage"  # 带建筑覆盖图
}
processed_dir = os.path.join(base_scene_path, "r_processed")
os.makedirs(os.path.join(base_scene_path, "r_output_images_dpm",output_dirs["default"]), exist_ok=True)
os.makedirs(os.path.join(base_scene_path, "r_output_images_dpm",output_dirs["building"]), exist_ok=True)

# 处理参数
start_num = 1
end_num = 700
cm_cell_size = (5.0, 5.0)
num_samples = int(20e6)
cell_size = 5  # 覆盖图单元格尺寸（米）
max_depth = 3


def safe_downsample(building_map, target_shape):
    """安全下采样建筑图到目标尺寸"""
    try:
        # 计算下采样比例
        scale_factor = building_map.shape[0] // target_shape[0]

        return block_reduce(
            building_map.astype(np.float32),
            block_size=(scale_factor, scale_factor),
            func=np.max
        )
    except Exception as e:
        print(f"下采样失败: {str(e)}")
        return None


def load_antenna_positions(city_num):
    """加载指定城市的天线坐标"""
    csv_path = os.path.join(
        processed_dir,
        "antenna_data",
        f"u{city_num}_antennas.csv",
    )
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def generate_building_overlay(city_num, cm_shape):
    """生成下采样后的建筑掩码"""
    building_path = os.path.join(
        processed_dir,
        "height_maps",
        f"u{city_num}.png"
    )

    if not os.path.exists(building_path):
        print(f"建筑图不存在: {building_path}")
        return None

    try:
        # 原始建筑图（1280x1280）
        building_map = np.flipud(imageio.imread(building_path))

        # 下采样到覆盖图分辨率
        building_ds = safe_downsample(building_map, cm_shape)
        if building_ds is None or building_ds.shape != cm_shape:
            print(f"下采样后尺寸不匹配: {building_ds.shape} vs {cm_shape}")
            return None

        return (building_ds > 0).astype(np.uint8) * 255  # 二值化掩码

    except Exception as e:
        print(f"生成建筑掩码失败: {str(e)}")
        return None


def process_transmitter(scene, tx_data, tx_idx, city_num):
    """处理单个发射机并保存两种覆盖图"""
    # 清理现有发射机
    for tx in list(scene.transmitters.keys()):
        scene.remove(tx)

    # 添加当前发射机
    tx = Transmitter(
        name=f"tx_{tx_idx}",
        position=[tx_data['x'], tx_data['y'], tx_data['z']],
        orientation=[0, 0, 0]
    )
    scene.add(tx)

    # 添加固定接收机
    if "rx" not in scene.receivers:
        rx = Receiver(
            name="rx",
            position=[45, 90, 1.5],
            orientation=[0, 0, 0]
        )
        scene.add(rx)

    # 生成覆盖图数据
    try:
        cm = scene.coverage_map(
            max_depth=max_depth,
            diffraction=True,
            cm_cell_size=cm_cell_size,
            num_samples=num_samples
        )
        pg_matrix = cm.path_gain[0].numpy()
    except Exception as e:
        print(f"覆盖图生成失败: {str(e)}")
        return False

    # 保存原始覆盖图
    cm.show(tx=0)
    default_path = os.path.join(
        base_scene_path,
        "r_output_images_dpm",
        output_dirs["default"],
        f"u{city_num}_{tx_idx:03d}.png"
    )
    # plt.savefig(default_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 生成带建筑的灰度图
    building_mask = generate_building_overlay(city_num, pg_matrix.shape)
    if building_mask is None:
        return False

    # 安全处理信号数据
    valid_mask = (pg_matrix > 1e-30) & (building_mask == 0)
    pg_db = np.full_like(pg_matrix, -np.inf)
    pg_db[valid_mask] = 10 * np.log10(pg_matrix[valid_mask])

    # 计算动态范围
    if np.any(valid_mask):
        min_db = np.min(pg_db[valid_mask])
        max_db = np.max(pg_db[valid_mask])
    else:
        min_db = max_db = -150

    # 生成灰度图像
    gray = np.zeros_like(pg_db, dtype=np.uint8)
    if max_db > min_db:
        norm = (pg_db - min_db) / (max_db - min_db)
        #gray[valid_mask] = (norm[valid_mask] * 255).astype(np.uint8)       #正常归一化
        gray[valid_mask] = 10 + (norm[valid_mask] * 240).astype(np.uint8)   #图像增强
    # 叠加建筑轮廓
    gray[building_mask > 0] = 255

    # 保存结果
    building_path = os.path.join(
        base_scene_path,
        "r_output_images_dpm",
        output_dirs["building"],
        f"{city_num}_{tx_idx:03d}.png"
    )
    imageio.imwrite(building_path, np.flipud(gray))

    return True


# 主处理流程保持不变，增加内存清理
def process_city(city_num):
    scene_path = os.path.join(base_scene_path, "rural",  f"u{city_num}", f"u{city_num}.xml")

    if not os.path.exists(scene_path):
        return

    scene = load_scene(scene_path)
    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
    scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "dipole", "cross")

    df = load_antenna_positions(city_num)
    if df is None:
        return

    success = 0
    for idx, row in df.iterrows():
        if process_transmitter(scene, row, idx, city_num):
            success += 1
            print(f"城市u{city_num}tx{idx + 1}完成")
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"城市u{city_num}完成: {success}/{len(df)}")


# scene_path = os.path.join(base_scene_path, "rural",  "u1", "u1.xml")
# scene = load_scene(scene_path)
# process_transmitter(scene, row, 22, 1)
for city_num in range(start_num, end_num+1):
	process_city(city_num)
