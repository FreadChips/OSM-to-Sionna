import requests
import random
import time
import math
import os
import xml.etree.ElementTree as ET

# 注册命名空间
ET.register_namespace('', "http://www.openstreetmap.org/osm/0.6")

API_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]


def get_building_count(bbox, endpoint_index=0):
    """查询给定bbox内的建筑数量（仅统计way和relation）"""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:25];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out count;
    """
    try:
        endpoint = API_ENDPOINTS[endpoint_index % len(API_ENDPOINTS)]
        response = requests.post(
            endpoint,
            data=query,
            headers={'Content-Type': 'text/plain'},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return int(data['elements'][0]['tags']['total'])
    except Exception as e:
        print(f"建筑数量查询失败: {str(e)}")
        return 0


def generate_valid_bbox(min_buildings, max_attempts=10, lat_range=(48.80, 48.90),
                        lon_range=(2.25, 2.45), area_size_km=1.28):
    """生成符合建筑数量要求的有效边界框"""
    attempts = 0
    while attempts < max_attempts:
        # 生成随机中心点
        center_lat = random.uniform(lat_range[0], lat_range[1])
        center_lon = random.uniform(lon_range[0], lon_range[1])

        # 计算边界框
        delta_lat = area_size_km / 111 /1.2 # 纬度跨度
        delta_lon = area_size_km / (111 * math.cos(math.radians(center_lat)))/1.5/1.2

        bbox = (
            center_lat - delta_lat / 2,
            center_lon - delta_lon / 2,
            center_lat + delta_lat / 2,
            center_lon + delta_lon / 2
        )

        # 检查建筑数量
        count = get_building_count(bbox, attempts % len(API_ENDPOINTS))
        if count >= min_buildings:
            print(f"发现有效区域：建筑数量 {count}，边界框 {bbox}")
            return bbox
        attempts += 1
        time.sleep(random.uniform(1, 3))  # 请求间隔

    print(f"在{max_attempts}次尝试后未找到符合条件的区域")
    return None


def download_osm_data(bbox, filename, endpoint_index=0):
    """下载指定区域的OSM数据并验证有效性"""
    south, west, north, east = bbox
    query = f"""
    [out:xml][timeout:45];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    try:
        endpoint = API_ENDPOINTS[endpoint_index % len(API_ENDPOINTS)]
        response = requests.post(
            endpoint,
            data=query,
            headers={'Content-Type': 'text/plain'},
            timeout=60
        )
        response.raise_for_status()

        # 保存前验证数据
        count = count_buildings_v2(response.content)
        if count == 0:
            print("警告：下载数据未包含任何建筑")

        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"数据下载失败: {str(e)}")
        return False


def count_buildings_v2(osm_content):
    """精确统计建筑数量"""
    try:
        root = ET.fromstring(osm_content)
        return len(root.findall('.//way/tag[@k="building"]/..')) + \
            len(root.findall('.//relation/tag[@k="building"]/..'))
    except Exception as e:
        print(f"XML解析错误: {str(e)}")
        return 0


if __name__ == "__main__":
    # 配置参数
    NUM_SAMPLES = 10  # 需要生成的数据集样本数
    MIN_BUILDINGS = 50  # 最小建筑数量阈值
    OUTPUT_DIR = "dataset"  # 输出目录
    MAX_ATTEMPTS = 15  # 每个样本的最大尝试次数

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_SAMPLES):
        print(f"\n=== 正在生成第 {i + 1}/{NUM_SAMPLES} 个样本 ===")

        # 生成有效边界框
        bbox = generate_valid_bbox(
            min_buildings=MIN_BUILDINGS,
            max_attempts=MAX_ATTEMPTS,
            lat_range=(48.80, 48.90),  # 巴黎纬度范围
            lon_range=(2.25, 2.45),  # 巴黎经度范围
            area_size_km=1.28
        )

        if not bbox:
            continue  # 跳过失败样本

        # 下载并保存数据
        filename = os.path.join(OUTPUT_DIR, f"sample_{i + 1}.osm")
        if download_osm_data(bbox, filename, endpoint_index=i):
            print(f"成功保存样本至 {filename}")
        else:
            print("样本下载失败")

        time.sleep(random.uniform(2, 5))  # 重要：遵守API使用规则



        #此文件生成数据地理范围会偏大，需要调整范围