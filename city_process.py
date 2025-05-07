import os
import glob
import csv
import random
import numpy as np
from plyfile import PlyData
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

# 配置参数
BASE_DIR = r"/home/lyus/fzq/mapss"  # 根目录
OUTPUT_ROOT = os.path.join(BASE_DIR, "u_processed")
CITY_PREFIX = "u"  # 城市目录前缀
OUTPUT_SUBDIRS = {  # 输出子目录结构
    "height_map": "height_maps",
    "antenna_map": "antenna_maps",
    "antenna_data": "antenna_data"
}


def setup_output_dirs():
    """创建城市输出目录结构"""
    for subdir in OUTPUT_SUBDIRS.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, subdir), exist_ok=True)
    # 创建统计文件
    stats_path = os.path.join(OUTPUT_ROOT, 'city_statistics.csv')
    if not os.path.exists(stats_path):
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['City', 'Avg Height (m)', 'Coverage (%)', 'Building Count'])

#通过读取ply文件计算覆盖率
def process_height_map2(city_meshes, city_name):
    """生成城市高度地图并返回统计信息"""
    img = Image.new('L', (1280, 1280), 0)
    draw = ImageDraw.Draw(img)
    ply_files = glob.glob(os.path.join(city_meshes, "Building_*.ply"))

    height_values = []
    buildings = []
    total_base_area = 0.0
    building_count = 0

    # 收集建筑数据
    for ply_path in ply_files:
        try:
            ply_data = PlyData.read(ply_path)
            vertices = ply_data['vertex'].data

            base_points = []
            max_height = 0.0
            for v in vertices:
                if abs(v['z']) < 1e-6:
                    base_points.append([v['x'], v['y']])
                if v['z'] > max_height:
                    max_height = v['z']

            if len(base_points) >= 3 and max_height > 0:
                points = np.array(base_points)
                hull = ConvexHull(points)
                buildings.append((points[hull.vertices], max_height))
                height_values.append(max_height)
                total_base_area += hull.volume  # 累加基底面积
                building_count += 1
        except Exception as e:
            print(f"Height map processing skipped {os.path.basename(ply_path)}: {str(e)}")

    # 计算统计指标
    stats = {
        'building_count': building_count,
        'avg_height': 0.0,
        'coverage': 0.0
    }

    if building_count > 0:
        stats['avg_height'] = np.mean(height_values)
        # 计算覆盖区域（地图范围1280x1280米）
        stats['coverage'] = (total_base_area / (1280 * 1280)) * 100  # 百分比

    # 绘制高度地图
    if building_count > 0:
        min_h, max_h = np.min(height_values), np.max(height_values)
        h_range = max_h - min_h if max_h > min_h else 1

        for hull_points, height in buildings:
            polygon = []
            for x, y in hull_points:
                px = np.clip(np.round(x + 640), 0, 1279).astype(int)
                py = np.clip(np.round(640 - y), 0, 1279).astype(int)
                polygon.append((px.item(), py.item()))

            if len(polygon) >= 3:
                gray = int(5 + 250 * (height - min_h) / h_range) if h_range > 0 else 255
                draw.polygon(polygon, fill=gray)

        output_path = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIRS["height_map"],
                                   f"{city_name}.png")
        img.save(output_path)
        return output_path, stats

    return None, stats

# 直接读取图像计算建筑覆盖率
def process_height_map(city_meshes, city_name):
    """生成城市高度地图并返回统计信息（优化版）"""
    img = Image.new('L', (1280, 1280), 0)
    draw = ImageDraw.Draw(img)
    ply_files = glob.glob(os.path.join(city_meshes, "Building_*.ply"))

    height_values = []
    building_count = 0

    # 第一阶段：收集建筑高度数据
    for ply_path in ply_files:
        try:
            ply_data = PlyData.read(ply_path)
            vertices = ply_data['vertex'].data

            base_points = []
            max_height = 0.0
            for v in vertices:
                if abs(v['z']) < 1e-6:
                    base_points.append([v['x'], v['y']])
                if v['z'] > max_height:
                    max_height = v['z']

            if len(base_points) >= 3 and max_height > 0:
                points = np.array(base_points)
                hull = ConvexHull(points)
                height_values.append(max_height)
                building_count += 1
        except Exception as e:
            print(f"Height map processing skipped {os.path.basename(ply_path)}: {str(e)}")

    # 统计指标初始化
    stats = {
        'building_count': building_count,
        'avg_height': 0.0,
        'coverage': 0.0
    }

    if building_count == 0:
        return None, stats

    # 绘制高度地图
    min_h, max_h = np.min(height_values), np.max(height_values)
    h_range = max_h - min_h if max_h > min_h else 1

    for ply_path in ply_files:  # 重新遍历确保数据一致性
        try:
            ply_data = PlyData.read(ply_path)
            vertices = ply_data['vertex'].data

            base_points = []
            max_height = 0.0
            for v in vertices:
                if abs(v['z']) < 1e-6:
                    base_points.append([v['x'], v['y']])
                if v['z'] > max_height:
                    max_height = v['z']

            if len(base_points) >= 3 and max_height > 0:
                points = np.array(base_points)
                hull = ConvexHull(points)

                polygon = []
                for x, y in points[hull.vertices]:
                    px = np.clip(np.round(x + 640), 0, 1279).astype(int)
                    py = np.clip(np.round(640 - y), 0, 1279).astype(int)
                    polygon.append((px.item(), py.item()))

                if len(polygon) >= 3:
                    gray = int(5 + 250 * (max_height - min_h) / h_range)
                    draw.polygon(polygon, fill=gray)
        except Exception as e:
            continue

    # 保存高度图
    output_path = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIRS["height_map"],
                               f"{city_name}.png")
    img.save(output_path)

    # 基于图像快速计算覆盖率
    height_array = np.array(img)
    building_pixels = np.count_nonzero(height_array)
    stats['coverage'] = (building_pixels / (1280 * 1280)) * 100
    stats['avg_height'] = np.mean(height_values)

    return output_path, stats


def save_city_stats(city_name, stats):
    """保存城市统计数据"""
    stats_path = os.path.join(OUTPUT_ROOT, 'city_statistics.csv')
    with open(stats_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            city_name,
            round(stats['avg_height'], 2),
            round(stats['coverage'], 2),
            stats['building_count']
        ])

def process_antennas(city_meshes, city_name, num_antennas=80):
    """带智能补全策略的天线生成函数"""
    # 第一阶段：收集建筑信息
    candidates = []
    ply_files = glob.glob(os.path.join(city_meshes, "Building_*.ply"))
    height_values = []  # 新增建筑高度收集器

    for ply_path in ply_files:
        try:
            ply_data = PlyData.read(ply_path)
            vertices = ply_data['vertex'].data

            base_points = []
            max_height = 0.0
            for v in vertices:
                if abs(v['z']) < 1e-6:
                    base_points.append([v['x'], v['y']])
                if v['z'] > max_height:
                    max_height = v['z']

            if len(base_points) >= 3 and max_height > 0:
                points = np.array(base_points)
                hull = ConvexHull(points)
                height_values.append(max_height)  # 记录建筑高度

                for x, y in points[hull.vertices]:
                    px = int(np.clip(np.round(x + 640), 0, 1279))
                    py = int(np.clip(np.round(640 - y), 0, 1279))

                    if 320 <= px <= 960 and 320 <= py <= 960:
                        candidates.append({
                            'x': x, 'y': y, 'z': max_height + 3,
                            'px': px, 'py': py,
                            'source': os.path.basename(ply_path)
                        })
        except Exception as e:
            print(f"Antenna processing skipped {os.path.basename(ply_path)}: {str(e)}")

    # 自动计算高度范围
    min_h = min(height_values) if height_values else 0
    max_h = max(height_values) if height_values else 0
    h_range = max_h - min_h if height_values else 1

    # 第二阶段：智能候选点补全
    # 从高度图自动获取建筑区域
    height_map_path = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIRS["height_map"],
                                   f"{city_name}.png")
    height_array = None
    if os.path.exists(height_map_path):
        height_img = Image.open(height_map_path).convert('L')
        height_array = np.array(height_img)

    selected = []
    remaining = num_antennas

    # 策略1：优先选择建筑顶点
    if candidates:
        select_num = min(len(candidates), remaining)
        selected = candidates[:select_num]
        remaining -= select_num

    # 策略2：建筑顶部补全
    if remaining > 0 and height_array is not None:
        building_pixels = np.argwhere(height_array > 0)
        if len(building_pixels) > 0:
            # 根据高度图灰度值计算实际高度
            gray_values = height_array[building_pixels[:, 0], building_pixels[:, 1]]
            actual_heights = (gray_values - 5) * h_range / 250 + min_h

            # 随机选择补全点
            indices = np.random.choice(len(building_pixels),
                                       size=min(remaining, len(building_pixels)),
                                       replace=False)
            for i, idx in enumerate(indices):
                py, px = building_pixels[idx]
                if 320 <= px <= 960 and 320 <= py <= 960:
                    selected.append({
                        'x': px - 640,
                        'y': 640 - py,
                        'z': actual_heights[i] + 3,  # 使用计算的实际高度
                        'px': px,
                        'py': py,
                        'source': 'roof_supplement',
                    })
                    remaining -= 1
                    if remaining <= 0:
                        break

    # 策略3：在地面补充随机点
    if remaining > 0:
        print(f"  Supplementing {remaining} antennas on ground")
        for _ in range(remaining):
            px = random.randint(320, 959)
            py = random.randint(320, 959)
            selected.append({
                'x': px - 640,
                'y': 640 - py,
                'z': 0.0,
                'px': px,
                'py': py,
                'source': 'ground_supplement',
            })

    # 生成输出文件
    csv_path = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIRS["antenna_data"],
                            f"{city_name}_antennas.csv")
    img_dir = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIRS["antenna_map"])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z', 'image_path', 'source'])

        for idx, ant in enumerate(selected[:num_antennas]):
            # 生成天线标记图
            img = Image.new('L', (1280, 1280), 0)
            draw = ImageDraw.Draw(img)
            draw.point((ant['px'], ant['py']), fill=255)

            img_name = f"{city_name}_{idx:03d}.png"
            img.save(os.path.join(img_dir, img_name))

            writer.writerow([
                ant['x'], ant['y'], ant['z'],
                os.path.join(OUTPUT_SUBDIRS["antenna_map"], img_name),
                ant['source']
            ])

    return csv_path, len(selected)

def process_city(city_dir):
    """处理单个城市"""
    city_name = os.path.basename(city_dir)
    print(f"\nProcessing city: {city_name}")
    city_meshes = os.path.join(city_dir, "meshes")

    if not os.path.exists(city_meshes):
        print(f"Missing meshes directory: {city_meshes}")
        return

    # 生成高度图并获取统计信息
    height_map, stats = process_height_map(city_meshes, city_name)
    if height_map:
        print(f"Height map generated: {height_map}")

    # 保存统计数据
    save_city_stats(city_name, stats)
    print(f"City statistics saved: {stats}")

    # 生成天线数据
    csv_path, num_ant = process_antennas(city_meshes, city_name)
    print(f"Generated {num_ant} antennas: {csv_path}")


def batch_process():
    """批量处理所有城市"""

    setup_output_dirs()

    city_dirs = glob.glob(os.path.join(BASE_DIR, "urban", f"{CITY_PREFIX}*"))

    for city_dir in city_dirs:
        if not os.path.isdir(city_dir):
            continue

        # 检查是否为有效城市目录
        if not os.path.exists(os.path.join(city_dir, "meshes")):
            print(f"Skipping invalid city directory: {city_dir}")
            continue

        try:
            process_city(city_dir)
        except Exception as e:
            print(f"Failed to process {city_dir}: {str(e)}")


if __name__ == "__main__":
    #batch_process()
    process_city("/home/lyus/fzq/mapss/urban/u498")
    print("\nBatch processing completed!")