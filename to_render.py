import sys 
import bpy
import os 
from mathutils import Vector,Matrix 
import json
import numpy as np 

#获得顶点坐标颜色
def get_vertex_texture_colors(mesh, texture_image_name):
    # 获取网格对象的UV层
    uv_layer = mesh.uv_layers.active.data
    
    # 获取纹理贴图
    texture_image = bpy.data.images.get(texture_image_name)
    if texture_image is None:
        print("Error: Texture image not found.")
        return None

    # 创建一个空的列表来存储每个顶点对应的纹理颜色
    vertex_colors = []

    # 遍历每个顶点的UV坐标，并根据UV坐标从纹理图像中获取颜色
    for loop in mesh.loops:
        uv_index = loop.vertex_index
        uv_coord = uv_layer[loop.index].uv
        
        # 将UV坐标转换为纹理图像的像素坐标
        tex_coord_x = int(uv_coord.x * texture_image.size[0])
        tex_coord_y = int(uv_coord.y * texture_image.size[1])
        
        # 获取纹理图像中对应位置的颜色
        tex_color = texture_image.pixels[(tex_coord_y * texture_image.size[0] + tex_coord_x) * texture_image.channels : 
                                          (tex_coord_y * texture_image.size[0] + tex_coord_x + 1) * texture_image.channels]
        
        # 将颜色添加到列表中
        vertex_colors.append(tex_color)

    # 将颜色数据转换为numpy数组
    vertex_colors_np = np.array(vertex_colors)

    return vertex_colors_np



#设置对象的材质
def set_emission_with_texture(obj_name, image_path):
    """
    为指定对象设置 Emission 材质，并将颜色设置为给定的图像纹理。

    Args:
    obj_name (str): 对象的名称。
    image_path (str): 图像文件的路径，用于纹理。
    """
    # 获取对象
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object named '{obj_name}' not found.")
        return

    # 确保对象有材质槽，如果没有则添加一个新的材质
    if not obj.data.materials:
        obj.data.materials.append(bpy.data.materials.new(name=f"{obj_name}_Material"))

    # 获取或创建材质
    mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 清空现有节点
    nodes.clear()

    # 添加 Emission 节点
    emission = nodes.new('ShaderNodeEmission')
    
    # 添加材质输出节点
    output = nodes.new('ShaderNodeOutputMaterial')
    
    # 连接 Emission 节点到材质输出
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # 添加图像纹理节点
    tex_image = nodes.new('ShaderNodeTexImage')
    # 加载图像文件
    tex_image.image = bpy.data.images.load(image_path, check_existing=True)

    # 连接图像纹理的颜色输出到 Emission 的颜色输入
    links.new(tex_image.outputs['Color'], emission.inputs['Color'])

#设置场景渲染模式
def set_view_shading(shading_type='RENDERED'):
    """
    设置当前3D视图的视图着色模式。
    
    Args:
    shading_type (str): 着色模式，可以是 'WIREFRAME', 'SOLID', 'MATERIAL', 'RENDERED' 等。
    """
    # 获取当前的3D视图区域
    for area in bpy.context.screen.areas:  # 遍历所有区域
        if area.type == 'VIEW_3D':  # 寻找3D视图区域
            # 获取3D视图的空间数据
            space = area.spaces.active
            # 设置视图着色模式
            space.shading.type = shading_type
            print(f"View shading set to {shading_type}")
            break
    else:
        print("No active 3D View found.")
#确认是否存在合集
def ensure_collection_exists(collection_name):
    """
    确保指定名字的集合存在于场景中。如果不存在，创建一个新的集合。
    
    Args:
    collection_name (str): 需要检查或创建的集合的名称。
    
    Returns:
    bpy.types.Collection: 存在或新创建的集合对象。
    """
    # 检查集合是否已存在
    if collection_name in bpy.data.collections: 
        return bpy.data.collections[collection_name]
    else:
        # 如果不存在，创建一个新的集合
        new_collection = bpy.data.collections.new(collection_name)
        # 将新集合添加到当前场景中
        bpy.context.scene.collection.children.link(new_collection)
        
        return new_collection
    
#保存为ply文件
def save_points_as_ply(points, file_path, comment="3D point cloud"):
    """
    Save a list of 3D points to a PLY file in ASCII format.
    
    Args:
    points (list of tuples): List of 3D points (x, y, z).
    file_path (str): Path to the output PLY file.
    comment (str): Comment to include in the header of the PLY file.
    """
    with open(file_path, 'w') as file:
        # 写入 PLY 头部
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"comment {comment}\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float red\n")
        file.write("property float green\n")
        file.write("property float blue\n") 
        file.write("property float nx\n")
        file.write("property float ny\n")
        file.write("property float nz\n") 
        file.write("property float index\n") 
        
        file.write("property float loc_x\n")
        file.write("property float loc_y\n")
        file.write("property float loc_z\n") 
        file.write("property float v1x\n")
        file.write("property float v1y\n")
        file.write("property float v1z\n") 
        file.write("property float v2x\n")
        file.write("property float v2y\n")
        file.write("property float v2z\n") 
        file.write("property float v3x\n")
        file.write("property float v3y\n")
        file.write("property float v3z\n")

        file.write("end_header\n")
        
        # 写入点数据
        for point in points:
            pass
            #print(point)
            file.write(f"{point}\n")
            #file.write(f"{point['global'][0]} {point['global'][1]} {point['global'][2]} {0} {0} {0} {1} {0} {0} {1999}\n")




 
#创建球体，分返回顶点
#细分次数 0：12 个顶点，20 个面
#细分次数 1：42 个顶点，80 个面
#细分次数 2：162 个顶点，320 个面
#细分次数 3：642 个顶点，1280 个面
#细分次数 4：2562 个顶点，5120 个面 
def create_sphere(radius, subdivisions):
    # 清除场景中的所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 添加细分球体
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, subdivisions=subdivisions)
    
    # 获取新创建的球体对象
    obj = bpy.context.object
    
    # 获取球体的所有顶点坐标
    vertices = [vert.co for vert in obj.data.vertices]


    #删除创建的对象
    # 确保在正确的上下文中操作，即确保在视图层中有对应对象
    bpy.context.view_layer.objects.active = obj
    # 选择对象
    obj.select_set(True)
    # 删除所有选中的对象
    bpy.ops.object.delete()

    return vertices


#创建灯光
def create_light(name,vertice,collection_name='lights'): 
 
    # 如果提供了集合名称，尝试添加到该集合，否则添加到主集合 
    collection=ensure_collection_exists(collection_name)
    # 创建一个新的点光源
    light_data = bpy.data.lights.new(name=name, type='POINT')
    light_object = bpy.data.objects.new(name="New_Point_Light", object_data=light_data)

    # 将灯光添加到场景中
    collection.objects.link(light_object)

    # 设置灯光的位置
    light_object.location = vertice

    # 设置灯光的能量和颜色
    light_data.energy = 100
    light_data.color = (1.0, 1.0, 1.0)  # 红色的光

    # 选择新创建的灯光并使其成为活动对象
    bpy.context.view_layer.objects.active = light_object
    light_object.select_set(True)



def create_camera(name, location, angle,collection_name='cameras'):
    """
    创建一个摄像机，并将其设置在指定位置，朝向原点 (0,0,0)。
    如果指定了集合名，摄像机将被添加到该集合中。
    
    Args:
    name (str): 摄像机的名称。
    location (tuple or list): 摄像机的位置坐标，格式为 (x, y, z)。
    collection_name (str, optional): 摄像机应添加到的集合的名称。如果没有指定，摄像机将被添加到场景的主集合中。
    """
    # 将位置转换为 Vector
    camera_location = Vector(location)
    
    # 创建摄像机数据
    camera_data = bpy.data.cameras.new(name=name)
    # 或者直接设置视场角（适用于透视摄像机）
    camera_data.angle = angle
    # 数值越大，图标显示越大
    camera_data.display_size = 1  

    # 创建摄像机对象
    camera_object = bpy.data.objects.new(name, camera_data)
    
    # 如果提供了集合名称，尝试添加到该集合，否则添加到主集合 
    collection=ensure_collection_exists(collection_name)
        
    # 将摄像机对象添加到集合中
    collection.objects.link(camera_object)
    
   
    

    
    center  = Vector((0, 0, shift_z))
    camera_location=camera_location + center
    direction =  camera_location- center
    
     # 设置摄像机的位置
    camera_object.location = camera_location

    # 定位摄像机指向原点
    camera_object.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    
    # 更新场景
    bpy.context.view_layer.update()
    return camera_object

#读取fbx文件，并创建一个自发光纹理贴图
def load_fbx(file_path,texture_path):
    
    fbx_path        = os.path.join(os.getcwd(), file_path)  
    texture_path        = os.path.join(os.getcwd(), texture_path) 
    # 导入FBX文件
    bpy.ops.import_scene.fbx(filepath=file_path)
    # 获取当前导入的对象
    imported_objects = bpy.context.selected_objects
 
    # 遍历每个导入的对象，设置纹理
    for obj in imported_objects: 
        if obj.type == 'MESH': 
            set_emission_with_texture(obj.name,texture_path)  
    return imported_objects

#设置场景
def set_world():

    # 获取当前场景的世界设置
    world = bpy.context.scene.world

    # 确保世界节点被激活
    world.use_nodes = True

    # 获取节点树和背景节点
    node_tree = world.node_tree
    nodes = node_tree.nodes
    
    # 创建一个新的背景节点和输出节点
    background_node = nodes.new(type='ShaderNodeBackground')
    output_node = nodes.new(type='ShaderNodeOutputWorld')

    # 设置背景颜色为纯白色
    background_node.inputs['Color'].default_value = (1, 1, 1, 1)  # RGBA，A通常不影响颜色

    # 将背景节点连接到输出节点
    node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

    # 确保使用节点来定义世界背景
    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    
    
    # 设置场景的 View Transform
    scene = bpy.context.scene
    scene.view_settings.view_transform = 'Standard'   
 

    # 设置渲染引擎和分辨率
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'  # 或者 'BLENDER_EEVEE'
    scene.render.resolution_x = int(ww)
    scene.render.resolution_y = int(hh)
    scene.render.resolution_percentage = 100

    # 设置输出格式为 PNG 
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'  # 包含透明通道
    scene.render.image_settings.compression = 0  # PNG 压缩等级


#渲染所有摄像机
def render_camera(camera,file_path): 
    # 循环遍历每个摄像机并渲染
    bpy.context.scene.camera = camera  # 设置当前摄像机 
    bpy.context.scene.render.filepath = file_path

    # 开始渲染并保存图像
    bpy.ops.render.render(write_still=True)
    print(f"Rendered and saved {file_path}")

def save_cameras_to_file(mesh_name='main', keyframe=0, is_render=True):
    #-----------------------------------------------------------------
    #输出数据# 确保输出目录存在
    output_folder=os.path.join(os.getcwd(), 'output') 
    images_folder=os.path.join(os.getcwd(), 'output/images') 
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)




    # 设置当前动画帧 
    bpy.context.scene.frame_set(keyframe)

    # 确保正确的对象和帧被选中
    bpy.context.view_layer.objects.active = bpy.data.objects[mesh_name]

    # 创建一个应用了所有修改器的 mesh 副本
    obj = bpy.context.object
    depsgraph = bpy.context.evaluated_depsgraph_get()  # 获取依赖图
    eval_obj = obj.evaluated_get(depsgraph)  # 获取应用了修改器后的对象
    mesh = eval_obj.to_mesh()  # 生成 mesh 数据

    mesh_data={}
    mesh_data['triangles']=[]
    # 输出三角面的位置信息
    for polygon in mesh.polygons:
        triangle={}
        triangle['index']=polygon.index  
        triangle['Vertices']=[] 
        for vert_idx in polygon.vertices:
            x=mesh.vertices[vert_idx].co.x
            y=mesh.vertices[vert_idx].co.y
            z=mesh.vertices[vert_idx].co.z
            triangle['Vertices'].append([x,y,z]) 

        mesh_data['triangles'].append(triangle)
 
 
    mesh_folder=os.path.join(os.getcwd(), 'output/meshes') 
    os.makedirs(mesh_folder, exist_ok=True)
    json_path_mesh= os.path.join(os.getcwd(), f'output/meshes/mesh_{keyframe}.json')  
    #保存mesh数据
    with open(json_path_mesh, "w") as file:
        json.dump(mesh_data, file)


    
    # 获取所有摄像机
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']

    #创建数据
    data_train={}
    data_train['camera_angle_x']=angle
    data_train['frames']=[]
    
    #生成测试用的数据
    data_test={}
    data_test['camera_angle_x']=angle
    data_test['frames']=[]

    index=0
    #生成训练用的数据
    for camera in cameras:
        frame={}
        frame['file_path']=f"images/{camera.name}"
        frame['transform_matrix']=np.array(camera.matrix_world).tolist()
        
        frame['keyframe']=keyframe #未来为了支持多关键帧动画生成预留的数据

        data_train['frames'].append(frame)

        if index==keyframe:
            data_test['frames'].append(frame)
 
        
        if is_render:
            render_camera(camera,os.path.join(output_folder, f"{frame['file_path']}.png"))

        index+=1

     
    
        
        


    json_path_train = os.path.join(os.getcwd(), 'output/transforms_train.json') 
    json_path_test  = os.path.join(os.getcwd(), 'output/transforms_test.json') 
    
    # 将数据保存到 JSON 文件中
    with open(json_path_train, "w") as file:
        json.dump(data_train, file)
    with open(json_path_test, "w") as file:
        json.dump(data_test, file)


#避免重复，更新数组
def update_array_efficient(old_set, new_array):
    for item in new_array:
        if item not in old_set:
            old_set.add(item)
    return old_set

'''

#进行三角形剖分
def subdivide_triangle(vertices, k):
    if k == 0:
        return []  # 返回空列表，避免重复添加初始顶点

    # 计算中点
    def midpoint(v1, v2):
        return ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2,(v1[2] + v2[2]) / 2)

    # 获取原始三角形的顶点
    p1, p2, p3 = vertices[0], vertices[1], vertices[2]
    m12 = midpoint(p1, p2)
    m23 = midpoint(p2, p3)
    m31 = midpoint(p3, p1)

    # 生成四个新的三角形
    new_triangles = [
        [p1, m12, m31],
        [p2, m23, m12],
        [p3, m31, m23],
        [m12, m23, m31]
    ]

    # 收集新生成的中点
    new_points = [m12, m23, m31]
     
    # 对每个新三角形递归细分
    for triangle in new_triangles:
        new_points.extend(subdivide_triangle(triangle, k - 1))

    return new_points

'''
#创建初始顶点
def create_points(k = 2 ): # 可以根据需要调整级别
    # 存储顶点坐标和面索引
    vertices = set()
    colors = []
    normals = []
 

    # 遍历每个导入的对象
    for obj in imported_objects:
        # 确保对象是网格类型
        if obj.type == 'MESH' and   obj.name == 'main':
            # 获取网格数据
            mesh = obj.data
            # 获取对象的世界变换矩阵
            world_matrix = obj.matrix_world

            # 遍历网格的每个面
            for index, face in enumerate(mesh.polygons):
                #获得三个顶点
                v1 = world_matrix @ mesh.vertices[face.vertices[0]].co
                v2 = world_matrix @ mesh.vertices[face.vertices[1]].co
                v3 = world_matrix @ mesh.vertices[face.vertices[2]].co

                # 计算局部坐标系
                o, n, x, y = calculate_local_coordinates(v1, v2, v3)

                # 执行递归的 Delaunay 三角剖分，级别设置为 k 
                subdivided_vertices = subdivide_triangle(v1, v2, v3, k)
                
                # 计算新顶点的局部坐标
                vertex_positions = calculate_local_position(subdivided_vertices, o, x, y, n,v1,v2,v3,index)
                #print(vertex_positions[0]['global'][0])
                update_array_efficient(vertices,vertex_positions)
                 
    return vertices
     





#创建三角形局部坐标系
def calculate_local_coordinates(v1, v2, v3):
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    # 计算中心点
    o = (v1 + v2 + v3) / 3
    # 计算法线 (使用叉积)
    n = np.cross(v2 - v1, v3 - v1)
    n = n / np.linalg.norm(n)  # 单位化
    # 定义X轴（从 o 指向 v1 的方向）
    x = v1 - o
    x = x / np.linalg.norm(x)  # 单位化
    # 定义Y轴（法线与X轴的叉积）
    y = np.cross(n, x)
    y = y / np.linalg.norm(y)  # 单位化
    return o, n, x, y

#递归细分三角形
def subdivide_triangle(v1, v2, v3,level):
    if level == 0:
        return [v1, v2, v3]
    else:
        # 计算每条边的中点
        m12 = (v1 + v2) / 2
        m23 = (v2 + v3) / 2
        m31 = (v3 + v1) / 2
        # 递归细分每个新生成的小三角形
        return (subdivide_triangle(v1, m12, m31, level-1) +
                subdivide_triangle(v2, m23, m12, level-1) +
                subdivide_triangle(v3, m31, m23, level-1) +
                subdivide_triangle(m12, m23, m31, level-1))
    
#计算本地位置
def calculate_local_position(vertices, o, x, y, n, v1, v2, v3,index):
    local_positions = []
    for v in vertices:
        v = np.array(v)
        # 计算局部坐标
        loc_x = np.dot(v - o, x)
        loc_y = np.dot(v - o, y)
        loc_z = np.dot(v - o, n)

        #x,y,z,r,g,b,nx,ny,nz,index,loc_x, loc_y, loc_z,v1x,v1y,v1z,v2x,v2y,v2z,v3x,v3y,v3z

        line  = "{:.4f} {:.4f} {:.4f} ".format(v[0],v[1],v[2])
        line += "{:.0f} {:.0f} {:.0f} ".format(0,0,0)
        line += "{:.4f} {:.4f} {:.4f} ".format(1,0,0)
        line += "{:.0f} ".format(index)
        line += "{:.4f} {:.4f} {:.4f} ".format(loc_x,loc_y,loc_z)
        line += "{:.4f} {:.4f} {:.4f} ".format(v1[0],v1[1],v1[2])
        line += "{:.4f} {:.4f} {:.4f} ".format(v2[0],v2[1],v2[2])
        line += "{:.4f} {:.4f} {:.4f} ".format(v3[0],v3[1],v3[2])
        

        local_positions.append(line)
        #local_positions.append("{
        #    'global': tuple(v),
        #    'local': (loc_x, loc_y, loc_z),
        #    'v1': tuple(v1),
        #    'v2': tuple(v2),
        #    'v3': tuple(v3)
        #})

    return local_positions













ww      =   1920/2
hh      =   1080/2   #生成图象尺寸
aa      =   65       #摄像机角度
shift_z =   0.8     #为了让模型位于摄像机中心，需要移动摄像机的位置
#主程序============================================================================
if __name__=="__main__":
    
    angle=aa * 3.1415926/180

    #设置世界和场景参数
    set_world()

    #创建球形点云作为摄像机初始位置
    vertices=create_sphere(3,3)
 
    #创建摄像机
    index=0
    for vertice in vertices:
        index=index+1
        #create_light(f'''light_{index}''',vertice)
        create_camera(f'''camera_{index}''',vertice,angle)
     
    #导入fbx
    imported_objects=load_fbx('data/main.fbx','data/main.png')  
    
 
  
    # 保存当前场景为 .blend 文件 
    bpy.ops.wm.save_as_mainfile(filepath= os.path.join(os.getcwd(), 'data/output.blend')  )
  
    #----------------------------
    #创建顶点
    vertices=create_points(k=2)
    #保存顶点
    ply_path = os.path.join(os.getcwd(), 'output/points3d.ply')
    save_points_as_ply(list(vertices),ply_path)

    

    #输出数据# 确保输出目录存在,参数是是否进行绘制
    save_cameras_to_file(keyframe=0,is_render=True)
 
    







    


    

    
  