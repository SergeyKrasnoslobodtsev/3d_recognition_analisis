import os, sys, math, numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Display.SimpleGui import init_display
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Graphic3d import Graphic3d_Camera, Graphic3d_NOM_PLASTIC, Graphic3d_MaterialAspect
from OCC.Core.Image import Image_AlienPixMap
from OCC.Core.gp import gp_Pnt, gp_Dir

SIZE = 524  # << целевой размер картинки

def fibonacci_sphere(samples=12, distance=5.0):
    pts = []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(samples):
        y = (1 - (i / float(samples - 1)) * 2) * distance
        r = max(1e-9, math.sqrt(distance * distance - y * y))
        th = phi * i
        x = math.cos(th) * r
        z = math.sin(th) * r
        pts.append((x, y, z))
    return pts

def read_step(path):
    r = STEPControl_Reader()
    assert r.ReadFile(path) == IFSelect_RetDone, f"STEP read failed: {path}"
    r.PrintCheckLoad(False, IFSelect_ItemsByEntity)
    r.PrintCheckTransfer(False, IFSelect_ItemsByEntity)
    r.TransferRoot(1)
    return r.Shape(1)

def save_view_524(view, out_path_png):
    """Жёстко сохраняем 524×524 без привязки к размеру окна."""
    pix = Image_AlienPixMap()
    # Если в вашей сборке нужно — раскомментируйте следующую строку:
    # view.MustBeResized()
    ok = view.ToPixMap(pix, SIZE, SIZE)
    assert ok, "ToPixMap failed"
    pix.Save(out_path_png)

def render_multiview_step(step_path: str, out_dir: str, views=12):
    shape = read_step(step_path)
    display, start_display, add_menu, add_func = init_display(display_triedron=False)

    # 1) Белый фон
    display.View.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))

    # 2) Создаём AIS и настраиваем материал: серый, без бликов
    ais = AIS_Shape(shape)
    ctx = display.Context
    ctx.SetColor(ais, Quantity_Color(0.65, 0.65, 0.65, Quantity_TOC_RGB), False)
    mat = Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC)
    mat.SetShininess(0.0)
    ctx.SetMaterial(ais, mat, False)
    ctx.Display(ais, True)

    # выключаем specular у шейдинга
    drw = ais.Attributes()                         # Handle_Prs3d_Drawer
    asp = drw.ShadingAspect().Aspect()            # Handle_Graphic3d_AspectFillArea3d

    drw.ShadingAspect().SetAspect(asp)

    # 3) Ортографическая камера
    cam = display.View.Camera()
    cam.SetProjectionType(Graphic3d_Camera.Projection_Orthographic)
    display.View.SetCamera(cam)
    display.View.FitAll()

    # Оцениваем дистанцию до центра
    cam = display.View.Camera()
    center = cam.Center()
    eye = cam.Eye()
    dist = math.sqrt((eye.X()-center.X())**2 + (eye.Y()-center.Y())**2 + (eye.Z()-center.Z())**2)

    # 4) Облёт по «золотой спирали»
    os.makedirs(out_dir, exist_ok=True)
    for i, (dx, dy, dz) in enumerate(fibonacci_sphere(views, dist)):
        cam = display.View.Camera()
        cam.SetEye(gp_Pnt(center.X()+dx, center.Y()+dy, center.Z()+dz))
        cam.SetCenter(center)
        cam.SetUp(gp_Dir(0, 1, 0))
        cam.SetProjectionType(Graphic3d_Camera.Projection_Orthographic)
        display.View.SetCamera(cam)
        display.View.FitAll()

        out_png = os.path.join(out_dir, f"mv_{i:03d}.png")  # PNG, sRGB
        save_view_524(display.View, out_png)

def make_multiview_dataset(models_dir_path, mvcnn_images_dir_path):
    for file in os.listdir(models_dir_path):
        if file.lower().endswith(".stp") or file.lower().endswith(".step"):
            in_path = os.path.join(models_dir_path, file)
            class_dir = os.path.join(mvcnn_images_dir_path, "default")
            render_multiview_step(in_path, class_dir, views=12)

# Пример:
make_multiview_dataset(r"D:\workspace\dataset\test", r"D:\workspace\dataset\multiview")
