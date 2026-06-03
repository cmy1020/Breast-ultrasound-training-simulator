import vtk

# ========= 配置 =========
STL_PATH = r"C:/Users/86150/Desktop/test.stl"  # 改成你的模型路径
PLANE_SIZE = 150.0      # 平面大小，应略大于模型 X/Y 范围
PLANE_RES = 200         # 2D 图像分辨率
INITIAL_Z = 0.0         # 初始切片位置
STEP_Z = 2.0            # 每次移动步长
# =======================


def load_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    print("STL loaded. Points:", polydata.GetNumberOfPoints(),
          "Cells:", polydata.GetNumberOfCells())
    return polydata


def center_polydata(polydata):
    bounds = [0] * 6
    polydata.GetBounds(bounds)
    print("Model bounds:", bounds)
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0
    cz = (bounds[4] + bounds[5]) / 2.0

    tf = vtk.vtkTransform()
    tf.Translate(-cx, -cy, -cz)

    tff = vtk.vtkTransformPolyDataFilter()
    tff.SetInputData(polydata)
    tff.SetTransform(tf)
    tff.Update()
    centered = tff.GetOutput()
    centered.GetBounds(bounds)
    print("Centered bounds:", bounds)
    return centered, bounds


def create_model_actor(polydata, color=(1.0, 0.75, 0.79)):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(0.4)
    return actor


def create_slice_plane_actor(z):
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(-PLANE_SIZE / 2, -PLANE_SIZE / 2, z)
    plane.SetPoint1(PLANE_SIZE / 2, -PLANE_SIZE / 2, z)
    plane.SetPoint2(-PLANE_SIZE / 2, PLANE_SIZE / 2, z)
    plane.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.2, 0.4, 0.9)
    actor.GetProperty().SetOpacity(0.3)
    return actor, plane


def update_plane_z(plane_source, new_z):
    plane_source.SetOrigin(-PLANE_SIZE / 2, -PLANE_SIZE / 2, new_z)
    plane_source.SetPoint1(PLANE_SIZE / 2, -PLANE_SIZE / 2, new_z)
    plane_source.SetPoint2(-PLANE_SIZE / 2, PLANE_SIZE / 2, new_z)
    plane_source.Update()


def compute_slice_image(polydata, z_position):
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(polydata)

    img = vtk.vtkImageData()
    img.SetDimensions(PLANE_RES, PLANE_RES, 1)
    spacing = PLANE_SIZE / PLANE_RES
    img.SetSpacing(spacing, spacing, 1.0)
    img.SetOrigin(-PLANE_SIZE / 2, -PLANE_SIZE / 2, z_position)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    inside_val = 220
    outside_val = 40

    for j in range(PLANE_RES):
        y = -PLANE_SIZE / 2 + (j + 0.5) * spacing
        for i in range(PLANE_RES):
            x = -PLANE_SIZE / 2 + (i + 0.5) * spacing
            d = implicit_distance.FunctionValue((x, y, z_position))
            val = inside_val if d < 0.0 else outside_val
            img.SetScalarComponentFromDouble(i, j, 0, 0, val)

    return img


class SliceKeyboardStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    用键盘↑↓控制切片位置：
      - ↑ : current_z += STEP_Z
      - ↓ : current_z -= STEP_Z
    """

    def __init__(self, model_polydata, plane_source, slice_viewer,
                 z0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_polydata = model_polydata
        self.plane_source = plane_source
        self.slice_viewer = slice_viewer
        self.current_z = z0

    def update_slice(self):
        print(f"update_slice: z = {self.current_z}")
        # 更新 3D 平面
        update_plane_z(self.plane_source, self.current_z)

        # 重新计算 2D 图像
        img = compute_slice_image(self.model_polydata, self.current_z)
        self.slice_viewer.SetInputData(img)
        self.slice_viewer.Render()

        # 重绘 3D
        self.GetInteractor().GetRenderWindow().Render()

    def OnKeyPress(self):
        key = self.GetInteractor().GetKeySym()
        if key == "Up":
            self.current_z += STEP_Z
            self.update_slice()
        elif key == "Down":
            self.current_z -= STEP_Z
            self.update_slice()

        # 其他按键保持 TrackballCamera 的默认行为
        super().OnKeyPress()


def main():
    # 1. 读取并居中 STL
    poly_raw = load_stl(STL_PATH)
    poly, bounds = center_polydata(poly_raw)

    global INITIAL_Z
    # 默认平面放在模型中心
    INITIAL_Z = (bounds[4] + bounds[5]) / 2.0
    print("Initial slice Z =", INITIAL_Z)

    # 2. 模型 actor
    model_actor = create_model_actor(poly)

    # 3. 初始平面 + 初始 2D 截面图像
    plane_actor, plane_source = create_slice_plane_actor(INITIAL_Z)
    slice_img = compute_slice_image(poly, INITIAL_Z)

    # ===== 3D 窗口 =====
    ren3d = vtk.vtkRenderer()
    ren3d.AddActor(model_actor)
    ren3d.AddActor(plane_actor)
    ren3d.SetBackground(0.1, 0.1, 0.1)
    ren3d.ResetCamera()

    win3d = vtk.vtkRenderWindow()
    win3d.AddRenderer(ren3d)
    win3d.SetSize(800, 600)
    win3d.SetWindowName("3D View (Use Up/Down keys)")

    iren3d = vtk.vtkRenderWindowInteractor()
    iren3d.SetRenderWindow(win3d)

    # ===== 2D 窗口 =====
    viewer2d = vtk.vtkImageViewer2()
    viewer2d.SetInputData(slice_img)
    viewer2d.SetColorWindow(255)
    viewer2d.SetColorLevel(127.5)

    win2d = vtk.vtkRenderWindow()
    win2d.SetSize(400, 400)
    win2d.SetWindowName("2D Slice View")
    viewer2d.SetRenderWindow(win2d)

    iren2d = vtk.vtkRenderWindowInteractor()
    viewer2d.SetupInteractor(iren2d)

    # ===== 键盘交互器绑定到 3D 窗口 =====
    style = SliceKeyboardStyle(
        model_polydata=poly,
        plane_source=plane_source,
        slice_viewer=viewer2d,
        z0=INITIAL_Z,
    )
    iren3d.SetInteractorStyle(style)

    # ===== 提示框（一次性的） =====
    msg = vtk.vtkTextActor()
    msg.SetInput("3D 窗口获取焦点后:\nUp / Down 调整切片位置")
    msgprop = msg.GetTextProperty()
    msgprop.SetFontSize(18)
    msgprop.SetColor(1, 1, 1)
    msg.SetDisplayPosition(10, 10)
    ren3d.AddActor2D(msg)

    win3d.Render()
    win2d.Render()

    iren3d.Initialize()
    iren2d.Initialize()

    iren3d.Start()
    iren2d.Start()


if __name__ == "__main__":
    main()