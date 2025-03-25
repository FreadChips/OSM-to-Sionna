import bpy
import bmesh
import xml.dom.minidom
import math
import mathutils
import os
import glob


# -------------------------------
# 1. 清空场景
# -------------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for scene in list(bpy.data.scenes):
        if scene != bpy.context.scene:
            bpy.data.scenes.remove(scene)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


# -------------------------------
# 2. 导入 OSM 文件并生成建筑（参数化文件路径）
# -------------------------------
def import_osm(osm_file_path):
    def get_or_create_material(name, diffuse_color):
        if name in bpy.data.materials:
            mat = bpy.data.materials[name]
        else:
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            for node in nodes:
                nodes.remove(node)
            node_out = nodes.new(type='ShaderNodeOutputMaterial')
            node_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
            node_diff.inputs['Color'].default_value = diffuse_color
            mat.node_tree.links.new(node_diff.outputs[0], node_out.inputs[0])
        return mat

    # 全局材质定义
    mat_wall_global = get_or_create_material("itu_marble", (1, 0.5, 0.2, 1))
    mat_roof_global = get_or_create_material("itu_metal", (0.29, 0.25, 0.21, 1))
    mat_concrete_global = get_or_create_material("itu_concrete", (0.5, 0.5, 0.5, 1))

    try:
        doc = xml.dom.minidom.parse(osm_file_path)
    except Exception as e:
        print(f"Error reading OSM file {osm_file_path}: {e}")
        return

    all_ways = doc.getElementsByTagName("way")
    buildings = []
    for el in all_ways:
        for ch in el.childNodes:
            if ch.attributes:
                if 'k' in ch.attributes.keys() and ch.attributes['k'].value == 'building':
                    buildings.append(el)
                    break

    nodes = doc.getElementsByTagName("node")
    id_to_tuple = {}
    for node in nodes:
        id_val = node.attributes['id'].value
        if node.hasAttribute('lon') and node.hasAttribute('lat'):
            lon = node.attributes['lon'].value
            lat = node.attributes['lat'].value
            id_to_tuple[id_val] = (lon, lat)

    all_buildings = []
    for b in buildings:
        lst = []
        nds = b.getElementsByTagName('nd')
        for ch in nds:
            if ch.tagName == 'nd':
                node_id = ch.attributes['ref'].value
                lst.append(id_to_tuple.get(node_id))
        tags = b.getElementsByTagName('tag')
        level = 1
        for tag in tags:
            if tag.tagName == 'tag' and tag.attributes['k'].value == 'building:levels':
                try:
                    level = int(tag.attributes['v'].value)
                except ValueError:
                    level = 1
        height = 20 if level == 1 else level * 5
        all_buildings.append((lst, height))

    all_lons = []
    all_lats = []
    for building in all_buildings:
        for coord in building[0]:
            lon, lat = coord
            all_lons.append(float(lon))
            all_lats.append(float(lat))
    center_lon = sum(all_lons) / len(all_lons)
    center_lat = sum(all_lats) / len(all_lats)

    def get_xy(lon, lat):
        lon = float(lon)
        lat = float(lat)
        mul = 111.321 * 1000
        diff_lon = lon - center_lon
        diff_lat = lat - center_lat
        return (diff_lon * mul, diff_lat * mul)

    buildings_xy = []
    for lst in all_buildings:
        tmp = []
        for i in lst[0]:
            tmp.append(get_xy(i[0], i[1]))
        h = lst[1]
        buildings_xy.append((tmp, h))

    all_x = [coord[0] for building in buildings_xy for coord in building[0]]
    all_y = [coord[1] for building in buildings_xy for coord in building[0]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    bbox_center_x = (min_x + max_x) / 2
    bbox_center_y = (min_y + max_y) / 2
    for i in range(len(buildings_xy)):
        adjusted_coords = []
        for (x, y) in buildings_xy[i][0]:
            adjusted_coords.append((x - bbox_center_x, y - bbox_center_y))
        buildings_xy[i] = (adjusted_coords, buildings_xy[i][1])

    scaling_factor = 1
    cnt = 0
    obs_list = []
    for lst in buildings_xy:
        cnt += 1
        tmp = []
        for i in lst[0]:
            x, y = i
            tmp.append((x * scaling_factor, y * scaling_factor, 0))
        h = lst[1]

        bm = bmesh.new()
        bm_verts = [bm.verts.new(v) for v in tmp]
        try:
            bm_face = bm.faces.new(bm_verts)
        except Exception as e:
            print(f"Face creation error: {e}")
            continue
        bm.normal_update()
        for f in bm.faces:
            if f.normal[2] < 0:
                f.normal_flip()

        me = bpy.data.meshes.new(f"Building_{cnt}")
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(f"Building_{cnt}", me)

        solidify = ob.modifiers.new("Solidify", type='SOLIDIFY')
        solidify.thickness = h * scaling_factor
        solidify.offset = 1

        ob.data.materials.clear()
        ob.data.materials.append(mat_wall_global)
        ob.data.materials.append(mat_roof_global)

        me.calc_normals_split()
        for poly in me.polygons:
            poly.material_index = 1 if poly.normal.z > 0 else 0

        obs_list.append(ob)
        print(f"Building {cnt} done!")

    for ob in obs_list:
        bpy.context.scene.collection.objects.link(ob)
    print("All buildings imported successfully!")

    bpy.ops.mesh.primitive_plane_add(size=1280, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "Ground_Plane"
    plane.data.materials.append(mat_concrete_global)
    print("Ground plane imported successfully!")


# -------------------------------
# 3. 裁剪超出区域的对象
# -------------------------------
def crop_scene():
    CENTER = mathutils.Vector((0, 0, 0))
    AREA_SIZE = (640, 640)
    AXIS = ('X', 'Y')

    def is_out_of_bounds(obj):
        global_verts = (obj.matrix_world @ v.co for v in obj.data.vertices)
        for v in global_verts:
            check_axes = [v.x, v.z] if AXIS == ('X', 'Z') else [v.x, v.y]
            center_axes = [CENTER.x, CENTER.z] if AXIS == ('X', 'Z') else [CENTER.x, CENTER.y]
            if (abs(check_axes[0] - center_axes[0]) > AREA_SIZE[0]) or \
                    (abs(check_axes[1] - center_axes[1]) > AREA_SIZE[1]):
                return True
        return False

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    deleted_count = 0
    for obj in mesh_objects:
        if is_out_of_bounds(obj):
            print(f"Deleting {obj.name} at {obj.location}")
            bpy.data.objects.remove(obj, do_unlink=True)
            deleted_count += 1
    print(f"Deleted {deleted_count} objects outside area")


# -------------------------------
# 4. 导出为 XML 文件（参数化导出路径）
# -------------------------------
def export_xml(export_filepath):
    bpy.ops.export_scene.mitsuba(
        filepath=export_filepath,
        axis_forward='Y',
        axis_up='Z',
        export_ids=True,
        use_selection=False,
        split_files=False,
        ignore_background=True
    )
    print(f"Scene exported to XML: {export_filepath}")


# -------------------------------
# 封装成单个文件的处理流程
# -------------------------------
def process_file(input_file, output_file):
    print(f"Processing input: {input_file}")
    clear_scene()
    import_osm(input_file)
    crop_scene()
    export_xml(output_file)
    print("-" * 50)


# -------------------------------
# 主流程：按序号处理输入目录中的所有 OSM 文件
# -------------------------------
def main():
    input_dir = "D:/Sionna/code/1catchdata/test/dataset/"
    output_base_dir = "D:/Sionna/maps"

    # 按文件名顺序获取所有 .osm 文件
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.osm")))
    if not input_files:
        print("未找到任何 OSM 文件！")
        return

    # 按序号（从1开始）遍历输入文件，生成输出文件夹和文件名
    for idx, input_file in enumerate(input_files, start=1):
        output_folder = os.path.join(output_base_dir, f"u{idx}")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"u{idx}.xml")
        process_file(input_file, output_file)


if __name__ == "__main__":
    main()
