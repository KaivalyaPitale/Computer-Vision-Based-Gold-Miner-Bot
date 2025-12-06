import os                         #$ Accessing folders/saving files
import glob                       #$ For quick finding recent images for a certain step
import json                       #$ Reading/Writing into JSON calibration file
import datetime                   #$ Generating date and time based timestamps while saving images
import threading                  #$ Running background threads like live alignment
import math                       #$ Calculations
import tkinter as tk              #$ GUI
from tkinter import messagebox    #$ Popups
import mss                        #$ Captuirng screenshots
import numpy as np
import cv2
from PIL import Image, ImageTk    #$ For converting to GUI friendly image format

# -------------------------------------------------
# Paths/constants
# -------------------------------------------------

CALIB_PATH="calibration/game_calibration.json"        
BACKGROUND_PATH="background.png"

#$ These values are generally constant or are with respect target patch
THRESHOLD_VALUE_FALLBACK=20
GROUND_LINE_Y=225
MIN_OBJECT_AREA=100

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def warmup_capture(): #$ This function prevents GUI size from changing after taking screenshots
    #$ Avoid DPI/resize jump on first grab
    _=mss.mss().monitors[0]


def load_calib():
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, "r") as f:
            return json.load(f)
    return {}


def save_calib(data):
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------
# Step 3: Deciding on distance of edges from target patch
# -------------------------------------------------

class CropDialog:
    LINE_THICK=1
    INNER_MARGIN=1

    def __init__(self, master, screenshots_dir, crops_dir):
        self.screenshots_dir=screenshots_dir
        self.crops_dir=crops_dir

        left_path=self._latest_with_key("hook_left")
        right_path=self._latest_with_key("hook_right")

        self.left_img=cv2.imread(left_path)
        self.right_img=cv2.imread(right_path)
        if self.left_img is None or self.right_img is None:
            raise RuntimeError("Missing hook_left or hook_right screenshots.")

        self.h, self.w=self.left_img.shape[:2]

        #$ initial crop: inset 10% from each side
        self.left_x=int(self.w*0.1)
        self.right_x=int(self.w*0.9)
        self.top_y=int(self.h*0.1)
        self.bottom_y=int(self.h*0.9)

        #$ stages: LEFT -> TOP -> RIGHT -> BOTTOM
        self.stage=0
        self.scale=0.5
        self.disp_w=int(self.w*self.scale)
        self.disp_h=int(self.h*self.scale)

        self.zoom_x=None
        self.zoom_y=None

        self.top=tk.Toplevel(master)
        self.top.title("Crop game window")

        self.canvas=tk.Canvas(self.top, width=self.disp_w, height=self.disp_h)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.stage_label=tk.Label(self.top, text="")
        self.stage_label.pack(pady=2)

        self.info_label=tk.Label(
            self.top,
            text="Click image to zoom. Use arrows to move current edge.\n"
                 "Order: LEFT → TOP → RIGHT → BOTTOM.\n"
                 "Crop is strictly inside the 1-pixel white lines.",
            justify="left"
        )
        self.info_label.pack(pady=2)

        self.btn_frame=tk.Frame(self.top)
        self.btn_frame.pack(pady=4)

        self.btns=[
            tk.Button(self.btn_frame, width=4),
            tk.Button(self.btn_frame, width=4),
            tk.Button(self.btn_frame, width=4),
            tk.Button(self.btn_frame, width=4),
        ]
        for b in self.btns:
            b.pack(side="left", padx=3)

        self.next_btn=tk.Button(self.top, text="Next/Crop", command=self.next_stage)
        self.next_btn.pack(fill="x", padx=8, pady=6)

        self.photo=None
        self.update_ui()

    def _latest_with_key(self, key):
        files=glob.glob(os.path.join(self.screenshots_dir, "*.png"))
        files=[f for f in files if key in os.path.basename(f)]
        files.sort(key=os.path.getmtime)
        if not files:
            raise RuntimeError(f"No screenshot found containing '{key}'")
        return files[-1]

    def img_with_line(self):
        img=self.left_img.copy()

        lx=max(0, min(self.w-1, self.left_x))
        rx=max(0, min(self.w-1, self.right_x))
        ty=max(0, min(self.h-1, self.top_y))
        by=max(0, min(self.h-1, self.bottom_y))

        t=self.LINE_THICK
        cv2.line(img, (lx, 0), (lx, self.h-1), (255, 255, 255), t)
        cv2.line(img, (rx, 0), (rx, self.h-1), (255, 255, 255), t)
        cv2.line(img, (0, ty), (self.w-1, ty), (255, 255, 255), t)
        cv2.line(img, (0, by), (self.w-1, by), (255, 255, 255), t)

        if self.zoom_x is not None and self.zoom_y is not None:
            img=self._draw_zoom(img)

        img=cv2.resize(img, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        return img

    def _draw_zoom(self, img):
	    #$ for finding precise game screen edge
        patch_size=30
        zoom_scale=8
        half=patch_size//2

        x0=max(0, self.zoom_x-half)
        y0=max(0, self.zoom_y-half)
        x1=min(self.w, self.zoom_x+half)
        y1=min(self.h, self.zoom_y+half)

        patch=img[y0:y1, x0:x1]
        if patch.size==0:
            return img

        patch_big=cv2.resize(
            patch,
            (patch.shape[1]*zoom_scale, patch.shape[0]*zoom_scale),
            interpolation=cv2.INTER_NEAREST,
        )
        ph, pw=patch_big.shape[:2]

        if self.stage==0:  # LEFT
            zx=0
            zy=max(0, min(self.h-ph, self.zoom_y-ph//2))
        elif self.stage==1:  # TOP
            zx=max(0, min(self.w-pw, self.zoom_x-pw//2))
            zy=0
        elif self.stage==2:  # RIGHT
            zx=self.w-pw
            zy=max(0, min(self.h-ph, self.zoom_y-ph//2))
        else:  # BOTTOM
            zx=max(0, min(self.w-pw, self.zoom_x-pw//2))
            zy=self.h-ph

        y2=min(self.h, zy+ph)
        x2=min(self.w, zx+pw)
        patch_big=patch_big[: y2-zy, : x2-zx]
        img[zy:y2, zx:x2]=patch_big
        cv2.rectangle(img, (zx, zy), (x2-1, y2-1), (0, 0, 255), 3)
        return img

    def update_canvas(self):
        img=self.img_with_line()
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        self.photo=ImageTk.PhotoImage(pil_img)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def update_ui(self):
        names=["LEFT edge", "TOP edge", "RIGHT edge", "BOTTOM edge"]
        self.stage_label.config(text=names[self.stage])

        if self.stage in (0, 2):
            cfg=[
                ("←",  lambda: self.shift(-1, 0)),
                ("⇐", lambda: self.shift(-15, 0)),
                ("→",  lambda: self.shift(1, 0)),
                ("⇒", lambda: self.shift(15, 0)),
            ]
        else:
            cfg=[
                ("↑",  lambda: self.shift(0, -1)),
                ("⇑", lambda: self.shift(0, -15)),
                ("↓",  lambda: self.shift(0, 1)),
                ("⇓", lambda: self.shift(0, 15)),
            ]

        for btn, (txt, cmd) in zip(self.btns, cfg):
            btn.config(text=txt, command=cmd)

        self.update_canvas()

    def on_canvas_click(self, event):
        ix=int(event.x/self.scale)
        iy=int(event.y/self.scale)
        self.zoom_x=max(0, min(self.w-1, ix))
        self.zoom_y=max(0, min(self.h-1, iy))
        self.update_canvas()

    def shift(self, dx, dy):
        if self.stage==0:
            if dx != 0:
                new=self.left_x+dx
                new=max(0, min(new, self.right_x-1))
                self.left_x=new
        elif self.stage==1:
            if dy != 0:
                new=self.top_y+dy
                new=max(0, min(new, self.bottom_y-1))
                self.top_y=new
        elif self.stage==2:
            if dx != 0:
                new=self.right_x+dx
                new=max(self.left_x+1, min(new, self.w-1))
                self.right_x=new
        elif self.stage==3:
            if dy != 0:
                new=self.bottom_y+dy
                new=max(self.top_y+1, min(new, self.h-1))
                self.bottom_y=new

        self.update_canvas()

    def next_stage(self):
        if self.stage < 3:
            self.stage += 1
            self.update_ui()
        else:
            self.apply_crop()
            self.top.destroy()

    def apply_crop(self):
        m=self.INNER_MARGIN

        x1_raw, x2_raw=self.left_x, self.right_x
        y1_raw, y2_raw=self.top_y, self.bottom_y

        x1=max(0, min(self.w-1, x1_raw+m))
        y1=max(0, min(self.h-1, y1_raw+m))
        x2=max(x1+1, min(self.w, x2_raw-m))
        y2=max(y1+1, min(self.h, y2_raw-m))

        crop_left=self.left_img[y1:y2, x1:x2]
        crop_right=self.right_img[y1:y2, x1:x2]

        os.makedirs(self.crops_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.crops_dir, "game_left.png"), crop_left)
        cv2.imwrite(os.path.join(self.crops_dir, "game_right.png"), crop_right)
        print("Saved game_left.png and game_right.png to", self.crops_dir)

        calib=load_calib()
        calib["game_crop"]={
            "left": int(x1),
            "top": int(y1),
            "right": int(x2),
            "bottom": int(y2),
        }
        calib["window_size"]=[int(x2-x1), int(y2-y1)]
        save_calib(calib)


# -------------------------------------------------
# Step 4: Target patch and centre selection
# -------------------------------------------------

class TargetDialog:
    def __init__(self, master, crops_dir):
        self.crops_dir=crops_dir

        img_path=self._latest_game_crop()
        self.img=cv2.imread(img_path)
        self.h, self.w=self.img.shape[:2]

        self.scale=0.5
        self.disp_w=int(self.w*self.scale)
        self.disp_h=int(self.h*self.scale)

        side0=int(min(self.w, self.h)*0.125)
        self.side=max(10, side0)
        self.cx=self.w//2
        self.cy=self.h//2
        self._clamp_center()

        self.top=tk.Toplevel(master)
        self.top.title("Select target square")

        self.canvas=tk.Canvas(self.top, width=self.disp_w, height=self.disp_h)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.info_label=tk.Label(
            self.top,
            text="Move the white square.\n+/++ /-/ -- change size.\n"
                 "Zoomed square at bottom-right.",
            justify="left"
        )
        self.info_label.pack(pady=2)

        ctrl_frame=tk.Frame(self.top)
        ctrl_frame.pack(pady=4)

        tk.Label(ctrl_frame, text="").grid(row=0, column=0)
        tk.Button(ctrl_frame, text="↑", width=3,
                  command=lambda: self.move(0, -1)).grid(row=0, column=1)
        tk.Button(ctrl_frame, text="⇑", width=3,
                  command=lambda: self.move(0, -15)).grid(row=0, column=2)
        tk.Label(ctrl_frame, text="").grid(row=0, column=3)

        tk.Button(ctrl_frame, text="⇐", width=3,
                  command=lambda: self.move(-15, 0)).grid(row=1, column=0)
        tk.Button(ctrl_frame, text="←", width=3,
                  command=lambda: self.move(-1, 0)).grid(row=1, column=1)
        tk.Button(ctrl_frame, text="→", width=3,
                  command=lambda: self.move(1, 0)).grid(row=1, column=2)
        tk.Button(ctrl_frame, text="⇒", width=3,
                  command=lambda: self.move(15, 0)).grid(row=1, column=3)

        tk.Label(ctrl_frame, text="").grid(row=2, column=0)
        tk.Button(ctrl_frame, text="↓", width=3,
                  command=lambda: self.move(0, 1)).grid(row=2, column=1)
        tk.Button(ctrl_frame, text="⇓", width=3,
                  command=lambda: self.move(0, 15)).grid(row=2, column=2)
        tk.Label(ctrl_frame, text="").grid(row=2, column=3)

        size_frame=tk.Frame(self.top)
        size_frame.pack(pady=4)
        tk.Button(size_frame, text="+", width=4,
                  command=lambda: self.resize(1)).pack(side="left", padx=2)
        tk.Button(size_frame, text="++", width=4,
                  command=lambda: self.resize(5)).pack(side="left", padx=2)
        tk.Button(size_frame, text="-", width=4,
                  command=lambda: self.resize(-1)).pack(side="left", padx=2)
        tk.Button(size_frame, text="--", width=4,
                  command=lambda: self.resize(-5)).pack(side="left", padx=2)

        self.save_btn=tk.Button(self.top, text="Save target", command=self.save_target)
        self.save_btn.pack(fill="x", padx=8, pady=6)

        self.photo=None
        self.update_canvas()

    def _latest_game_crop(self):
        left=os.path.join(self.crops_dir, "game_left.png")
        right=os.path.join(self.crops_dir, "game_right.png")
        if os.path.exists(left):
            return left
        return right

    def _clamp_center(self):
        half=self.side//2
        self.cx=max(half, min(self.w-half-1, self.cx))
        self.cy=max(half, min(self.h-half-1, self.cy))

    def img_with_square(self):
        img=self.img.copy()

        half=self.side//2
        x1=self.cx-half
        y1=self.cy-half
        x2=self.cx+half
        y2=self.cy+half

        x1=max(0, x1)
        y1=max(0, y1)
        x2=min(self.w, x2)
        y2=min(self.h, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 4)

        cross_len=8
        cv2.line(img, (self.cx-cross_len, self.cy),
                 (self.cx+cross_len, self.cy), (255, 255, 255), 1)
        cv2.line(img, (self.cx, self.cy-cross_len),
                 (self.cx, self.cy+cross_len), (255, 255, 255), 1)

        img=self._draw_zoom(img)
        img=cv2.resize(img, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        return img

    def _draw_zoom(self, img):
        half=self.side//2
        x1=max(0, self.cx-half)
        y1=max(0, self.cy-half)
        x2=min(self.w, self.cx+half)
        y2=min(self.h, self.cy+half)

        square_w=max(1, x2-x1)
        square_h=max(1, y2-y1)

        patch=img[y1:y2, x1:x2]
        if patch.size==0:
            return img

        max_scale_w=self.w/square_w
        max_scale_h=self.h/square_h
        zoom_scale=int(min(4, max_scale_w, max_scale_h))
        if zoom_scale < 1:
            zoom_scale=1

        patch_big=cv2.resize(
            patch,
            (square_w*zoom_scale, square_h*zoom_scale),
            interpolation=cv2.INTER_NEAREST,
        )
        ph, pw=patch_big.shape[:2]

        zx=self.w-pw
        zy=self.h-ph

        y2p=min(self.h, zy+ph)
        x2p=min(self.w, zx+pw)
        patch_big=patch_big[: y2p-zy, : x2p-zx]
        img[zy:y2p, zx:x2p]=patch_big
        cv2.rectangle(img, (zx, zy), (x2p-1, y2p-1), (0, 0, 255), 3)
        return img

    def update_canvas(self):
        img=self.img_with_square()
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        self.photo=ImageTk.PhotoImage(pil_img)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def move(self, dx, dy):
        self.cx += dx
        self.cy += dy
        self._clamp_center()
        self.update_canvas()

    def resize(self, dsize):
        self.side=max(10, self.side+dsize)
        self._clamp_center()
        self.update_canvas()

    def on_canvas_click(self, event):
        ix=int(event.x/self.scale)
        iy=int(event.y/self.scale)
        self.cx, self.cy=ix, iy
        self._clamp_center()
        self.update_canvas()

    def save_target(self):
        half=self.side//2
        x1=max(0, self.cx-half)
        y1=max(0, self.cy-half)
        x2=min(self.w, self.cx+half)
        y2=min(self.h, self.cy+half)

        calib=load_calib()
        calib["target_center"]=[int(self.cx), int(self.cy)]
        calib["target_size"]=int(x2-x1)
        save_calib(calib)

        os.makedirs(self.crops_dir, exist_ok=True)
        crop=self.img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(self.crops_dir, "target.png"), crop)

        print("Saved target_center, target_size and target.png")
        self.top.destroy()
# -------------------------------------------------
# Step 5: Hook centre and patch selection
# -------------------------------------------------

class HookPatchDialog:

    def __init__(self, master, crops_dir):
        self.crops_dir=crops_dir

        img_path=self._latest_game_crop()
        self.img=cv2.imread(img_path)
        if self.img is None:
            raise RuntimeError("Could not load cropped game image for hook patch.")

        self.h, self.w=self.img.shape[:2]

        self.scale=0.5
        self.disp_w=int(self.w*self.scale)
        self.disp_h=int(self.h*self.scale)

        # base patch height; width=2*height
        base=int(min(self.w, self.h)*0.2)
        self.height=max(20, base)

        calib=load_calib()
        default_offset=float(calib.get("hook_center_offset_ratio", 0.05))
        self.offset_ratio=max(0.0, min(0.5, default_offset))

        # hook center (crosshair)
        self.cx=self.w//2
        self.cy=self.h//2
        self._clamp_center()

        self.top=tk.Toplevel(master)
        self.top.title("Select hook patch")

        self.canvas=tk.Canvas(self.top, width=self.disp_w, height=self.disp_h)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.info_label=tk.Label(
            self.top,
            text="Move the white rectangle over the idle hook.\n"
                 "Aspect ratio is fixed at 2:1 (width=2 × height).\n"
                 "Hook center (crosshair) is offset from the top by a percentage of height.\n"
                 "+/++ /-/ -- change patch height.\n"
                 "Use the slider to adjust hook-center offset.\n"
                 "Zoomed patch is shown at bottom-right.",
            justify="left"
        )
        self.info_label.pack(pady=2)

        # movement controls
        ctrl_frame=tk.Frame(self.top)
        ctrl_frame.pack(pady=4)

        tk.Label(ctrl_frame, text="").grid(row=0, column=0)
        tk.Button(ctrl_frame, text="↑", width=3,
                  command=lambda: self.move(0, -1)).grid(row=0, column=1)
        tk.Button(ctrl_frame, text="⇑", width=3,
                  command=lambda: self.move(0, -15)).grid(row=0, column=2)
        tk.Label(ctrl_frame, text="").grid(row=0, column=3)

        tk.Button(ctrl_frame, text="⇐", width=3,
                  command=lambda: self.move(-15, 0)).grid(row=1, column=0)
        tk.Button(ctrl_frame, text="←", width=3,
                  command=lambda: self.move(-1, 0)).grid(row=1, column=1)
        tk.Button(ctrl_frame, text="→", width=3,
                  command=lambda: self.move(1, 0)).grid(row=1, column=2)
        tk.Button(ctrl_frame, text="⇒", width=3,
                  command=lambda: self.move(15, 0)).grid(row=1, column=3)

        tk.Label(ctrl_frame, text="").grid(row=2, column=0)
        tk.Button(ctrl_frame, text="↓", width=3,
                  command=lambda: self.move(0, 1)).grid(row=2, column=1)
        tk.Button(ctrl_frame, text="⇓", width=3,
                  command=lambda: self.move(0, 15)).grid(row=2, column=2)
        tk.Label(ctrl_frame, text="").grid(row=2, column=3)

        # size controls (height only; width follows)
        size_frame=tk.Frame(self.top)
        size_frame.pack(pady=4)
        tk.Button(size_frame, text="+", width=4,
                  command=lambda: self.resize(2)).pack(side="left", padx=2)
        tk.Button(size_frame, text="++", width=4,
                  command=lambda: self.resize(8)).pack(side="left", padx=2)
        tk.Button(size_frame, text="-", width=4,
                  command=lambda: self.resize(-2)).pack(side="left", padx=2)
        tk.Button(size_frame, text="--", width=4,
                  command=lambda: self.resize(-8)).pack(side="left", padx=2)

        # hook-center offset slider
        offset_frame=tk.Frame(self.top)
        offset_frame.pack(pady=2, fill="x")
        tk.Label(offset_frame, text="Hook center offset (% of patch height from top):")\
            .pack(anchor="w")
        self.offset_var=tk.DoubleVar(value=self.offset_ratio*100.0)
        self.offset_slider=tk.Scale(
            offset_frame,
            from_=0,
            to=40,
            resolution=1,
            orient="horizontal",
            length=220,
            command=self.on_offset_change,
        )
        self.offset_slider.set(self.offset_var.get())
        self.offset_slider.pack(anchor="w")

        self.save_btn=tk.Button(self.top, text="Save hook patch", command=self.save_hook_patch)
        self.save_btn.pack(fill="x", padx=8, pady=6)

        self.photo=None
        self.update_canvas()

    # ---------- geometry helpers ----------

    def _latest_game_crop(self):
        left=os.path.join(self.crops_dir, "game_left.png")
        right=os.path.join(self.crops_dir, "game_right.png")
        if os.path.exists(left):
            return left
        return right

    def _clamp_size(self):
        #$ ensure height fits image with width=2*height
        h=int(self.height)
        if h < 10:
            h=10
        max_by_width=self.w//2
        max_by_height=self.h
        h=min(h, max_by_width, max_by_height)
        self.height=h

    def _clamp_center(self):
        self._clamp_size()
        #$ keep offset_ratio in [0, 0.5]
        self.offset_ratio=max(0.0, min(0.5, float(self.offset_ratio)))

        half_w=self.height              #$ width=2*height -> half width=height
        top_margin=self.offset_ratio*self.height
        bottom_margin=(1.0-self.offset_ratio)*self.height

        cx_min=half_w
        cx_max=self.w-half_w-1

        cy_min=int(round(top_margin))
        cy_max=int(round(self.h-bottom_margin))

        self.cx=max(cx_min, min(cx_max, self.cx))
        self.cy=max(cy_min, min(cy_max, self.cy))

    def _current_rect(self):
        h=int(self.height)
        w=2*h
        r=float(self.offset_ratio)

        #$ vertical: top based on hook center and offset
        top_f=self.cy-r*h
        top=int(round(top_f))
        top=max(0, min(self.h-h, top))
        bottom=top+h

        #$ horizontal: symmetric around hook center
        half_w=w//2
        left=self.cx-half_w
        left=max(0, min(self.w-w, left))
        right=left+w

        return int(left), int(top), int(right), int(bottom)

    # ---------- drawing ----------

    def img_with_square(self):
        img=self.img.copy()

        x1, y1, x2, y2=self._current_rect()

        #$ outer rectangle (patch)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 4)

        #$ crosshair at hook center
        cross_len=8
        cv2.line(img, (self.cx-cross_len, self.cy),
                 (self.cx+cross_len, self.cy), (255, 255, 255), 1)
        cv2.line(img, (self.cx, self.cy-cross_len),
                 (self.cx, self.cy+cross_len), 1)

        img=self._draw_zoom(img)
        img=cv2.resize(img, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        return img

    def _draw_zoom(self, img):
        x1, y1, x2, y2=self._current_rect()

        w_rect=max(1, x2-x1)
        h_rect=max(1, y2-y1)

        patch=img[y1:y2, x1:x2]
        if patch.size==0:
            return img

        max_scale_w=self.w/w_rect
        max_scale_h=self.h/h_rect
        zoom_scale=int(min(4, max_scale_w, max_scale_h))
        if zoom_scale < 1:
            zoom_scale=1

        patch_big=cv2.resize(
            patch,
            (int(w_rect*zoom_scale), int(h_rect*zoom_scale)),
            interpolation=cv2.INTER_NEAREST,
        )
        ph, pw=patch_big.shape[:2]

        zx=self.w-pw
        zy=self.h-ph

        y2p=min(self.h, zy+ph)
        x2p=min(self.w, zx+pw)
        patch_big=patch_big[: y2p-zy, : x2p-zx]
        img[zy:y2p, zx:x2p]=patch_big
        cv2.rectangle(img, (zx, zy), (x2p-1, y2p-1), (0, 0, 255), 3)
        return img

    def update_canvas(self):
        img=self.img_with_square()
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        self.photo=ImageTk.PhotoImage(pil_img)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    # ---------- interactions ----------

    def move(self, dx, dy):
        self.cx += dx
        self.cy += dy
        self._clamp_center()
        self.update_canvas()

    def resize(self, dheight):
        self.height=max(10, self.height+dheight)
        self._clamp_center()
        self.update_canvas()

    def on_canvas_click(self, event):
        ix=int(event.x/self.scale)
        iy=int(event.y/self.scale)
        self.cx, self.cy=ix, iy
        self._clamp_center()
        self.update_canvas()

    def on_offset_change(self, value):
        try:
            v=float(value)
        except ValueError:
            return
        self.offset_ratio=max(0.0, min(0.5, v/100.0))
        self.offset_var.set(v)
        self._clamp_center()
        self.update_canvas()

    # ---------- save ----------

    def save_hook_patch(self):
        x1, y1, x2, y2=self._current_rect()
        width=x2-x1
        height=y2-y1

        calib=load_calib()
        #$ hook center (crosshair position)
        calib["hook_center"]=[int(self.cx), int(self.cy)]
        calib["hook_patch_width"]=int(width)
        calib["hook_patch_height"]=int(height)
        calib["hook_center_offset_ratio"]=float(self.offset_ratio)
        save_calib(calib)

        os.makedirs(self.crops_dir, exist_ok=True)
        crop=self.img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(self.crops_dir, "hook_patch.png"), crop)

        print(
            "Saved hook_center, hook_patch_width, hook_patch_height, "
            f"hook_center_offset_ratio={self.offset_ratio:.3f}, and hook_patch.png"
        )
        self.top.destroy()



# -------------------------------------------------
# Step 6: Merge LEFT/RIGHT hooks into background
# -------------------------------------------------

class MergeBackgroundDialog:
    SCALE=0.5

    def __init__(self, master, crops_dir):
        self.crops_dir=crops_dir

        left_path=os.path.join(crops_dir, "game_left.png")
        right_path=os.path.join(crops_dir, "game_right.png")

        self.left_img=cv2.imread(left_path)
        self.right_img=cv2.imread(right_path)

        if self.left_img is None or self.right_img is None:
            self.top=tk.Toplevel(master)
            self.top.title("Merge error")
            tk.Label(self.top, text="Missing game_left.png/game_right.png").pack(padx=8, pady=8)
            tk.Button(self.top, text="Close", command=self.top.destroy).pack(pady=4)
            print("[Merge] game_left.png/game_right.png missing.")
            return

        if self.left_img.shape != self.right_img.shape:
            h=min(self.left_img.shape[0], self.right_img.shape[0])
            w=min(self.left_img.shape[1], self.right_img.shape[1])
            self.left_img=cv2.resize(self.left_img, (w, h))
            self.right_img=cv2.resize(self.right_img, (w, h))

        self.h, self.w=self.left_img.shape[:2]
        self.split_x=self.w//2
        self.zoom_x=self.split_x
        self.zoom_y=self.h//2

        self.top=tk.Toplevel(master)
        self.top.title("Merge hook LEFT/RIGHT into background")

        self.disp_w=int(self.w*self.SCALE)
        self.disp_h=int(self.h*2*self.SCALE)

        self.canvas=tk.Canvas(self.top, width=self.disp_w, height=self.disp_h)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        info=(
            "Top: preview with visible seam.\n"
            "Bottom: merged background (no line).\n"
            "Use ← ⇐ → ⇒ to move seam; click to move zoom window."
        )
        tk.Label(self.top, text=info, justify="left").pack(pady=4)

        btn_frame=tk.Frame(self.top)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="←", width=4,
                  command=lambda: self.shift_split(-1)).pack(side="left", padx=2)
        tk.Button(btn_frame, text="⇐", width=4,
                  command=lambda: self.shift_split(-15)).pack(side="left", padx=2)
        tk.Button(btn_frame, text="→", width=4,
                  command=lambda: self.shift_split(1)).pack(side="left", padx=2)
        tk.Button(btn_frame, text="⇒", width=4,
                  command=lambda: self.shift_split(15)).pack(side="left", padx=2)

        tk.Button(self.top, text="Save as background.png", command=self.save_background)\
            .pack(fill="x", padx=8, pady=6)

        self.photo=None
        self.update_canvas()

    def _make_view_image(self):
        x=max(1, min(self.w-2, self.split_x))

        preview=np.zeros_like(self.left_img)
        preview[:, :x]=self.right_img[:, :x]
        preview[:, x:]=self.left_img[:, x:]
        cv2.line(preview, (x-1, 0), (x-1, self.h-1), (255, 255, 255), 1)

        merged=np.zeros_like(self.left_img)
        merged[:, :x]=self.right_img[:, :x]
        merged[:, x:]=self.left_img[:, x:]

        view=np.zeros((self.h*2, self.w, 3), dtype=np.uint8)
        view[0:self.h, :, :]=preview
        view[self.h:2*self.h, :, :]=merged

        #$ zoom from preview
        if self.zoom_x is not None and self.zoom_y is not None:
            zx=max(0, min(self.w-1, self.zoom_x))
            zy=max(0, min(self.h-1, self.zoom_y))
            patch_size=30
            half=patch_size//2

            x0=max(0, zx-half)
            y0=max(0, zy-half)
            x1=min(self.w, zx+half)
            y1=min(self.h, zy+half)

            patch=preview[y0:y1, x0:x1]
            if patch.size > 0:
                zoom_scale=8
                patch_big=cv2.resize(
                    patch,
                    ((x1-x0)*zoom_scale, (y1-y0)*zoom_scale),
                    interpolation=cv2.INTER_NEAREST,
                )
                ph, pw=patch_big.shape[:2]
                hx=max(0, self.w-pw)
                hy=0
                y2=min(self.h, hy+ph)
                x2=min(self.w, hx+pw)
                patch_big=patch_big[: y2-hy, : x2-hx]
                view[hy:y2, hx:x2]=patch_big
                cv2.rectangle(view, (hx, hy), (x2-1, y2-1), (0, 0, 255), 2)

        return view

    def update_canvas(self):
        img=self._make_view_image()
        img=cv2.resize(img, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        self.photo=ImageTk.PhotoImage(pil_img)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def shift_split(self, dx):
        self.split_x=max(1, min(self.w-2, self.split_x+dx))
        self.update_canvas()

    def on_canvas_click(self, event):
        vx=int(event.x/self.SCALE)
        vy=int(event.y/self.SCALE)
        if vy >= self.h:
            vy -= self.h
        self.zoom_x=max(0, min(self.w-1, vx))
        self.zoom_y=max(0, min(self.h-1, vy))
        self.update_canvas()

    def save_background(self):
        x=max(1, min(self.w-2, self.split_x))
        merged=np.zeros_like(self.left_img)
        merged[:, :x]=self.right_img[:, :x]
        merged[:, x:]=self.left_img[:, x:]

        cv2.imwrite(BACKGROUND_PATH, merged)
        print(f"[Merge] Saved merged background to {BACKGROUND_PATH}")
        self.top.destroy()


# -------------------------------------------------
# Step 7: Live alignment
# -------------------------------------------------

def live_align_loop(crops_dir, calib, align_state):
    target_path=os.path.join(crops_dir, "target.png")
    target_img=cv2.imread(target_path)
    if target_img is None:
        print("[Align] target.png not found; cannot run live alignment.")
        return

    if "target_center" not in calib or "target_size" not in calib or "window_size" not in calib:
        print("[Align] Missing target_center/target_size/window_size in calibration.")
        return

    cx, cy=calib["target_center"]
    target_size=calib["target_size"]
    win_w, win_h=calib["window_size"]

    half=target_size//2
    target_box_tl_x=cx-half
    target_box_tl_y=cy-half
    offset_x=-target_box_tl_x
    offset_y=-target_box_tl_y

    with mss.mss() as sct:
        monitor=sct.monitors[0]
        print("[Align] Live alignment started. Close or click Confirm to stop.")
        while not align_state["stop"]:
            grab=sct.grab(monitor)
            frame=np.array(grab)[:, :, :3]
            fh, fw=frame.shape[:2]

            res=cv2.matchTemplate(frame, target_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc=cv2.minMaxLoc(res)
            tx, ty=max_loc

            if max_val < 0.6:
                disp=cv2.resize(frame, (fw//4, fh//4))
                cv2.putText(disp, "Waiting for target...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                gx=tx+offset_x
                gy=ty+offset_y
                gx=max(0, min(gx, fw-win_w))
                gy=max(0, min(gy, fh-win_h))
                game=frame[gy:gy+win_h, gx:gx+win_w].copy()
                disp=cv2.resize(game, (win_w//4, win_h//4))
                cv2.rectangle(disp, (0, 0), (disp.shape[1]-1, disp.shape[0]-1), (0, 255, 0), 1)
                cv2.putText(disp, f"score {max_val:.2f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow("Live Game Window", disp)
            if cv2.waitKey(1) & 0xFF==27:
                break

        cv2.destroyWindow("Live Game Window")
        print("[Align] Live alignment stopped.")


# -------------------------------------------------
# Step 9: TNT threshold calibration (selected TNT cause it can split into two object sometimes)
# -------------------------------------------------

class TNTThresholdDialog:

    def __init__(self, master, screenshots_dir, crops_dir):
        self.screenshots_dir=screenshots_dir
        self.crops_dir=crops_dir
        self.top=tk.Toplevel(master)
        self.top.title("TNT threshold calibration")

        self.calib=load_calib()
        if "game_crop" not in self.calib or "window_size" not in self.calib:
            tk.Label(self.top, text="Missing game_crop/window_size in calibration.").pack(padx=8, pady=8)
            tk.Button(self.top, text="Close", command=self.top.destroy).pack(pady=4)
            return

        if not os.path.exists(BACKGROUND_PATH):
            tk.Label(self.top, text="Missing background.png, run merge step first.").pack(padx=8, pady=8)
            tk.Button(self.top, text="Close", command=self.top.destroy).pack(pady=4)
            return

        bg_full=cv2.imread(BACKGROUND_PATH)
        win_w, win_h=self.calib["window_size"]
        self.bg_game=cv2.resize(bg_full, (win_w, win_h))

        self.game_crop_rect=self.calib["game_crop"]
        self.win_w, self.win_h=win_w, win_h

        self.game_img=None 
        self.h=self.w=None

        self.patch_coords=None  #$(x1,y1,x2,y2)
        self.patch_game=None
        self.patch_bg=None

        self.thresh_var=tk.IntVar(value=int(self.calib.get("threshold_value", THRESHOLD_VALUE_FALLBACK)))

        # 1) screen selection stage
        self._build_screen_selection()

    def _latest_level_paths(self):
        all_paths=[
            p for p in glob.glob(os.path.join(self.screenshots_dir, "*.png"))
            if "levels" in os.path.basename(p)
        ]
        all_paths.sort(key=os.path.getmtime)
        #$ only last 6
        return all_paths[-6:]

    def _build_screen_selection(self):
        for w in self.top.winfo_children():
            w.destroy()

        self.top.title("TNT threshold: select screenshot with TNT")

        paths=self._latest_level_paths()
        if not paths:
            tk.Label(self.top, text="No level screenshots found (step 8).").pack(padx=8, pady=8)
            tk.Button(self.top, text="Close", command=self.top.destroy).pack(pady=4)
            return

        self.screen_paths=paths
        self.screen_photos=[]
        self.screen_labels=[]
        self.selected_index=None

        tk.Label(self.top, text="Select the screenshot that contains TNT, then click 'Next'.")\
            .pack(pady=4)

        grid_frame=tk.Frame(self.top)
        grid_frame.pack(padx=8, pady=4)

        crop=self.game_crop_rect
        gx1, gy1, gx2, gy2=crop["left"], crop["top"], crop["right"], crop["bottom"]
        thumb_w=self.win_w//8
        thumb_h=self.win_h//8

        for i, p in enumerate(paths):
            img_full=cv2.imread(p)
            if img_full is None:
                continue
            game=img_full[gy1:gy2, gx1:gx2]
            if game.shape[1] != self.win_w or game.shape[0] != self.win_h:
                game=cv2.resize(game, (self.win_w, self.win_h))

            thumb=cv2.resize(game, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
            img_rgb=cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            pil_img=Image.fromarray(img_rgb)
            photo=ImageTk.PhotoImage(pil_img)
            self.screen_photos.append(photo)

            row=i//2
            col=i % 2
            lbl=tk.Label(grid_frame, image=photo, bd=2, relief="flat")
            lbl.grid(row=row, column=col, padx=5, pady=5)
            lbl.bind("<Button-1>", lambda e, idx=i: self._select_screen(idx))
            self.screen_labels.append(lbl)

        btn_frame=tk.Frame(self.top)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Next", command=self._go_to_patch_stage)\
            .pack(side="left", padx=4)
        tk.Button(btn_frame, text="Cancel", command=self.top.destroy)\
            .pack(side="left", padx=4)

    def _select_screen(self, idx):
        self.selected_index=idx
        for i, lbl in enumerate(self.screen_labels):
            if i==idx:
                lbl.config(relief="solid", bd=3, bg="#00aa00")
            else:
                lbl.config(relief="flat", bd=2, bg=self.top.cget("bg"))

    def _go_to_patch_stage(self):
        if self.selected_index is None:
            messagebox.showinfo("TNT threshold", "Please select a screenshot that contains TNT.")
            return

        # prepare chosen game image
        crop=self.game_crop_rect
        gx1, gy1, gx2, gy2=crop["left"], crop["top"], crop["right"], crop["bottom"]
        img_full=cv2.imread(self.screen_paths[self.selected_index])
        game=img_full[gy1:gy2, gx1:gx2]
        if game.shape[1] != self.win_w or game.shape[0] != self.win_h:
            game=cv2.resize(game, (self.win_w, self.win_h))

        self.game_img=game
        self.h, self.w=self.game_img.shape[:2]

        # initial TNT box: somewhere reasonable (center)
        side0=int(min(self.w, self.h)*0.2)
        self.side=max(20, side0)
        self.cx=self.w//2
        self.cy=self.h//2
        self._clamp_center()

        self._build_patch_and_threshold_ui()

    def _clamp_center(self):
        if self.game_img is None:
            return
        half=self.side//2
        self.cx=max(half, min(self.w-half-1, self.cx))
        self.cy=max(half, min(self.h-half-1, self.cy))

    def _build_patch_and_threshold_ui(self):
        for w in self.top.winfo_children():
            w.destroy()

        self.top.title("TNT threshold: select TNT and tune threshold")

        #$ main game canvas with selection box
        self.display_scale=0.6
        disp_w=int(self.w*self.display_scale)
        disp_h=int(self.h*self.display_scale)

        main_frame=tk.Frame(self.top)
        main_frame.pack(padx=8, pady=8)

        left_col=tk.Frame(main_frame)
        left_col.pack(side="left", padx=4)

        tk.Label(left_col, text="Game frame (move box onto TNT):")\
            .pack(anchor="w")
        self.game_canvas=tk.Canvas(left_col, width=disp_w, height=disp_h, bg="black")
        self.game_canvas.pack()
        self.game_canvas.bind("<Button-1>", self._on_game_click)

        #$ control buttons for moving/resizing box
        ctrl=tk.Frame(left_col)
        ctrl.pack(pady=4)
        tk.Button(ctrl, text="↑", width=3,
                  command=lambda: self._move_box(0, -1)).grid(row=0, column=1)
        tk.Button(ctrl, text="⇑", width=3,
                  command=lambda: self._move_box(0, -15)).grid(row=0, column=2)

        tk.Button(ctrl, text="⇐", width=3,
                  command=lambda: self._move_box(-15, 0)).grid(row=1, column=0)
        tk.Button(ctrl, text="←", width=3,
                  command=lambda: self._move_box(-1, 0)).grid(row=1, column=1)
        tk.Button(ctrl, text="→", width=3,
                  command=lambda: self._move_box(1, 0)).grid(row=1, column=2)
        tk.Button(ctrl, text="⇒", width=3,
                  command=lambda: self._move_box(15, 0)).grid(row=1, column=3)

        tk.Button(ctrl, text="↓", width=3,
                  command=lambda: self._move_box(0, 1)).grid(row=2, column=1)
        tk.Button(ctrl, text="⇓", width=3,
                  command=lambda: self._move_box(0, 15)).grid(row=2, column=2)

        size_frame=tk.Frame(left_col)
        size_frame.pack(pady=2)
        tk.Button(size_frame, text="+", width=4,
                  command=lambda: self._resize_box(2)).pack(side="left", padx=4)
        tk.Button(size_frame, text="-", width=4,
                  command=lambda: self._resize_box(-2)).pack(side="left", padx=4)

        #$ right column: TNT patch/background/binary
        right_col=tk.Frame(main_frame)
        right_col.pack(side="left", padx=4)

        tk.Label(right_col, text="TNT patch/Background/Binary mask")\
            .pack(anchor="w")

        self.preview_canvas=tk.Canvas(
            right_col,
            width=3*120+40,
            height=120+40,
            bg="black"
        )
        self.preview_canvas.pack(pady=2)

        slider_frame=tk.Frame(right_col)
        slider_frame.pack(pady=4, fill="x")
        tk.Label(slider_frame, text="Background-subtraction threshold:")\
            .pack(anchor="w")
        self.slider=tk.Scale(
            slider_frame,
            from_=5,
            to=80,
            resolution=1,
            orient="horizontal",
            variable=self.thresh_var,
            length=250,
            command=lambda v: self._update_preview()
        )
        self.slider.pack(anchor="w")

        self.obj_count_var=tk.StringVar(value="Object count: -")
        tk.Label(right_col, textvariable=self.obj_count_var)\
            .pack(anchor="w", pady=2)

        tk.Label(
            self.top,
            text="Adjust the threshold so TNT appears as one solid object in the binary mask.\n"
                 "Then click 'Save threshold'.",
            justify="left"
        ).pack(pady=4)

        btn_frame=tk.Frame(self.top)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Save threshold", command=self._save_threshold)\
            .pack(side="left", padx=4)
        tk.Button(btn_frame, text="Cancel", command=self.top.destroy)\
            .pack(side="left", padx=4)

        self.photo_game_disp=None
        self.photo_patches=[]

        self._draw_game_canvas()
        self._update_preview()

    def _current_box_coords(self):
        half=self.side//2
        x1=max(0, self.cx-half)
        y1=max(0, self.cy-half)
        x2=min(self.w, self.cx+half)
        y2=min(self.h, self.cy+half)
        return x1, y1, x2, y2

    def _draw_game_canvas(self):
        disp=cv2.resize(
            self.game_img,
            (int(self.w*self.display_scale), int(self.h*self.display_scale)),
            interpolation=cv2.INTER_NEAREST
        )
        disp_rgb=cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        #$ draw box on downscaled image
        x1, y1, x2, y2=self._current_box_coords()
        sx=int(x1*self.display_scale)
        sy=int(y1*self.display_scale)
        ex=int(x2*self.display_scale)
        ey=int(y2*self.display_scale)
        cv2.rectangle(disp_rgb, (sx, sy), (ex, ey), (255, 255, 255), 2)

        pil_img=Image.fromarray(disp_rgb)
        self.photo_game_disp=ImageTk.PhotoImage(pil_img)
        self.game_canvas.delete("all")
        self.game_canvas.create_image(0, 0, anchor="nw", image=self.photo_game_disp)

    def _update_preview(self):
        x1, y1, x2, y2=self._current_box_coords()
        if x2 <= x1 or y2 <= y1:
            return

        self.patch_game=self.game_img[y1:y2, x1:x2]
        self.patch_bg=self.bg_game[y1:y2, x1:x2]

        t=self.thresh_var.get()

        diff=cv2.absdiff(self.patch_game, self.patch_bg)
        gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary=cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        #$ object count in patch
        contours, _=cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count=0
        for c in contours:
            if cv2.contourArea(c) >= MIN_OBJECT_AREA:
                count += 1
        self.obj_count_var.set(f"Object count: {count} (threshold={t})")

        #$ draw three patches resized to 120x120 area each
        self.preview_canvas.delete("all")
        self.photo_patches=[]

        patches=[
            ("TNT patch", self.patch_game),
            ("Background", self.patch_bg),
            ("Binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
        ]
        tile_size=120
        x_start=10
        for i, (name, im) in enumerate(patches):
            ph, pw=im.shape[:2]
            scale=min(tile_size/pw, tile_size/ph)
            new_w=max(1, int(pw*scale))
            new_h=max(1, int(ph*scale))
            im_resized=cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            canvas_img=np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            x_off=(tile_size-new_w)//2
            y_off=(tile_size-new_h)//2
            canvas_img[y_off:y_off+new_h, x_off:x_off+new_w]=im_resized
            cv2.rectangle(canvas_img, (0, 0), (tile_size-1, tile_size-1), (255, 255, 255), 1)

            img_rgb=cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
            pil_img=Image.fromarray(img_rgb)
            photo=ImageTk.PhotoImage(pil_img)
            self.photo_patches.append(photo)

            px=x_start+i*(tile_size+10)
            self.preview_canvas.create_image(px, 10, anchor="nw", image=photo)
            self.preview_canvas.create_text(
                px+tile_size//2,
                10+tile_size+12,
                text=name,
                fill="white"
            )

    def _move_box(self, dx, dy):
        self.cx += dx
        self.cy += dy
        self._clamp_center()
        self._draw_game_canvas()
        self._update_preview()

    def _resize_box(self, dside):
        self.side=max(20, self.side+dside)
        self._clamp_center()
        self._draw_game_canvas()
        self._update_preview()

    def _on_game_click(self, event):
        #$ move center to click
        ix=int(event.x/self.display_scale)
        iy=int(event.y/self.display_scale)
        self.cx, self.cy=ix, iy
        self._clamp_center()
        self._draw_game_canvas()
        self._update_preview()

    def _save_threshold(self):
        t=int(self.thresh_var.get())
        calib=load_calib()
        calib["threshold_value"]=t
        save_calib(calib)
        print(f"[TNT] Saved threshold_value={t} into {CALIB_PATH}")
        self.top.destroy()


# -------------------------------------------------
# Step 10: Captcha style labeling
# -------------------------------------------------

class LabelingCaptchaDialog:

    TILE_SIZE=80

    def __init__(self, master, level_paths, calib, crops_dir):
        self.calib=calib
        self.crops_dir=crops_dir
        self.level_paths=level_paths

        self.objects=self._collect_objects()
        if not self.objects:
            self.top=tk.Toplevel(master)
            self.top.title("Label objects")
            tk.Label(self.top, text="No objects found for labeling.").pack(padx=8, pady=8)
            tk.Button(self.top, text="Close", command=self.top.destroy).pack(pady=4)
            return

        self.n=len(self.objects)
        #$ sort indices by area desc
        self.order=sorted(
            range(self.n),
            key=lambda i: self.objects[i]["props"]["area"],
            reverse=True
        )
        self.labels=[None]*self.n  # None, 'trash', or real label

        #$ label sequence (trash first)
        self.label_sequence=[
            "trash",
            "gold_nugget",
            "diamond",
            "bone",
            "skull",
            "mole_left",
            "mole_right",
            "mole_diamond_left",
            "mole_diamond_right",
            "stone",
            "gold_small",
            "rock",
            "bag",
            "tnt",
            "gold_medium",
            "gold_large",
        ]

        self.current_label_index=0
        self.current_label=self.label_sequence[0]

        self.current_candidates=[]
        self.current_page=0
        self.selected_tiles=set()  #$ tile indices 0 through 15 on current page
        self.tile_indices=[None]*16  #$ map tile slot -> object index
        self.tile_photos=[None]*16

        # UI
        self.top=tk.Toplevel(master)
        self.top.title("Label objects (captcha mode)")

        self.header_label=tk.Label(self.top, text="", font=("Segoe UI", 10, "bold"))
        self.header_label.pack(pady=4)

        self.skip_label_btn=tk.Button(self.top, text="Skip this label", command=self._skip_current_label)
        self.skip_label_btn.pack(pady=(0, 4))

        grid_frame=tk.Frame(self.top)
        grid_frame.pack(padx=8, pady=4)
        self.tile_labels=[]
        for r in range(4):
            for c in range(4):
                idx=r*4+c
                lbl=tk.Label(grid_frame, bd=1, relief="solid")
                lbl.grid(row=r, column=c, padx=3, pady=3)
                lbl.bind("<Button-1>", lambda e, t=idx: self._on_tile_click(t))
                self.tile_labels.append(lbl)

        self.page_label=tk.Label(self.top, text="")
        self.page_label.pack(pady=2)

        btn_row=tk.Frame(self.top)
        btn_row.pack(pady=4)
        self.next_skip_btn=tk.Button(btn_row, text="Skip page", command=self._next_or_skip_page)
        self.next_skip_btn.pack(side="left", padx=4)
        tk.Button(btn_row, text="Finish labeling", command=self._finish).pack(side="left", padx=4)

        # label count table (two rows)
        table_frame=tk.Frame(self.top)
        table_frame.pack(pady=4, padx=8, fill="x")
        tk.Label(table_frame, text="Label counts (click to jump):").pack(anchor="w")

        self.label_table_frame=tk.Frame(table_frame)
        self.label_table_frame.pack(fill="x", pady=(2, 0))

        self.label_buttons={}  # label -> tk.Button
        self._build_label_table()

        self._start_label(self.current_label)

    def _collect_objects(self):
        objs=[]
        if not os.path.exists(BACKGROUND_PATH):
            print("[Labeling] No background.png found; skipping object collection.")
            return objs

        bg=cv2.imread(BACKGROUND_PATH)
        if "window_size" not in self.calib or "game_crop" not in self.calib:
            print("[Labeling] Missing window_size/game_crop in calibration.")
            return objs

        win_w, win_h=self.calib["window_size"]
        game_crop=self.calib["game_crop"]
        gx1, gy1, gx2, gy2=game_crop["left"], game_crop["top"], game_crop["right"], game_crop["bottom"]

        bg_resized=cv2.resize(bg, (win_w, win_h))
        threshold_value=int(self.calib.get("threshold_value", THRESHOLD_VALUE_FALLBACK))

        for lvl_idx, path in enumerate(self.level_paths):
            img_full=cv2.imread(path)
            if img_full is None:
                continue
            aligned=img_full[gy1:gy2, gx1:gx2]
            if aligned.shape[1] != win_w or aligned.shape[0] != win_h:
                aligned=cv2.resize(aligned, (win_w, win_h))

            diff=cv2.absdiff(aligned, bg_resized)
            gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, binary=cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            if GROUND_LINE_Y < binary.shape[0]:
                binary[:GROUND_LINE_Y, :]=0

            contours, _=cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area < MIN_OBJECT_AREA:
                    continue
                x, y, w, h=cv2.boundingRect(cnt)
                mask=np.zeros(aligned.shape[:2], np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                roi=aligned[y:y+h, x:x+w]
                mask_roi=mask[y:y+h, x:x+w]
                mean_bgr=cv2.mean(roi, mask=mask_roi)[:3]

                props={
                    "area": float(area),
                    "avg_b": mean_bgr[0],
                    "avg_g": mean_bgr[1],
                    "avg_r": mean_bgr[2],
                    "aspect_ratio": (w/h if h>0 else 0.0),
                }
                patch=aligned[y:y+h, x:x+w].copy()
                objs.append({"patch": patch, "props": props})

        print(f"[Labeling] Collected {len(objs)} objects from {len(self.level_paths)} level screenshots.")
        return objs

    # ----- Label table -----

    def _build_label_table(self):
        # two rows of labels
        row1=["gold_nugget", "diamond", "bone", "skull",
                "mole_left", "mole_right", "mole_diamond_left", "mole_diamond_right"]
        row2=["stone", "gold_small", "rock", "bag",
                "tnt", "gold_medium", "gold_large"]

        # 'trash' row
        trash_row=tk.Frame(self.label_table_frame)
        trash_row.pack(fill="x")
        tk.Label(trash_row, text="trash:", width=7, anchor="w")\
            .pack(side="left")
        btn=tk.Button(trash_row, text="0", width=4,
                        command=lambda l="trash": self._jump_to_label(l))
        btn.pack(side="left")
        self.label_buttons["trash"]=btn

        def make_row(labels):
            row=tk.Frame(self.label_table_frame)
            row.pack(fill="x", pady=2)
            for lab in labels:
                btn=tk.Button(
                    row,
                    text=f"{lab}:0",
                    width=16,
                    command=lambda l=lab: self._jump_to_label(l)
                )
                btn.pack(side="left", padx=2)
                self.label_buttons[lab]=btn

        make_row(row1)
        make_row(row2)

    def _update_label_counts_and_colors(self):
        counts={}
        for lab in self.labels:
            if lab:
                counts[lab]=counts.get(lab, 0)+1

        for lab, btn in self.label_buttons.items():
            c=counts.get(lab, 0)
            if lab=="trash":
                btn.config(text=str(c))
            else:
                btn.config(text=f"{lab}:{c}")

            if c==0:
                color="#cc0000"   # red
            elif c==1:
                color="#ff8800"   # orange
            elif c==2:
                color="#ffcc00"   # yellow
            elif c==3:
                color="#aadd44"   # light green
            else:
                color="#00aa00"   # green
            btn.config(bg=color, activebackground=color)

    # ----- Label navigation -----

    def _start_label(self, label):
        self.current_label=label
        self.current_label_index=self.label_sequence.index(label)
        self.selected_tiles.clear()
        self.current_page=0

        # candidates for this label
        if label=="trash":
            self.current_candidates=[i for i in self.order if self.labels[i] is None]
        else:
            self.current_candidates=[
                i for i in self.order
                if self.labels[i] is None  # ignore trash and already-labeled
            ]

        self.header_label.config(text=f"Label: {label} — select all matching objects")
        self._update_label_counts_and_colors()
        self._update_grid()

    def _skip_current_label(self):
        self._goto_next_label()

    def _goto_next_label(self):
        idx=self.current_label_index+1
        while idx < len(self.label_sequence):
            lab=self.label_sequence[idx]
            # see if there is anything left for this label
            if lab=="trash":
                candidates=[i for i in self.order if self.labels[i] is None]
            else:
                candidates=[i for i in self.order if self.labels[i] is None]
            if candidates:
                self._start_label(lab)
                return
            idx += 1
        # no more labels with candidates so finish finish
        self._finish()

    def _jump_to_label(self, label):
        # clicking in label table jumps there
        self._start_label(label)

    # ----- Grid/tile logic -----

    def _update_grid(self):
        # if no candidates for this label, go to next label
        if not self.current_candidates:
            self._goto_next_label()
            return

        total_pages=(len(self.current_candidates)+15)//16
        if self.current_page >= total_pages:
            self._goto_next_label()
            return

        start=self.current_page*16
        end=min(len(self.current_candidates), start+16)
        page_indices=self.current_candidates[start:end]

        self.page_label.config(
            text=f"Page {self.current_page+1}/{total_pages}  "
                 f"(objects {start+1}-{end} of {len(self.current_candidates)})"
        )

        self.tile_indices=[None]*16
        self.tile_photos=[None]*16
        self.selected_tiles=set()  # reset when entering a new page

        for t in range(16):
            lbl=self.tile_labels[t]
            if t < len(page_indices):
                obj_idx=page_indices[t]
                self.tile_indices[t]=obj_idx
                patch=self.objects[obj_idx]["patch"]
                tile_img=self._make_tile_image(patch, selected=False)
                img_rgb=cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                pil_img=Image.fromarray(img_rgb)
                photo=ImageTk.PhotoImage(pil_img)
                self.tile_photos[t]=photo
                lbl.config(image=photo)
            else:
                # empty tile=black
                empty=np.zeros((self.TILE_SIZE, self.TILE_SIZE, 3), dtype=np.uint8)
                cv2.rectangle(empty, (0, 0), (self.TILE_SIZE-1, self.TILE_SIZE-1), (80, 80, 80), 1)
                img_rgb=cv2.cvtColor(empty, cv2.COLOR_BGR2RGB)
                pil_img=Image.fromarray(img_rgb)
                photo=ImageTk.PhotoImage(pil_img)
                self.tile_photos[t]=photo
                lbl.config(image=photo)
                self.tile_indices[t]=None

        self._update_next_skip_button()

    def _make_tile_image(self, patch, selected):
        th=self.TILE_SIZE
        tw=self.TILE_SIZE
        canvas=np.zeros((th, tw, 3), dtype=np.uint8)

        ph, pw=patch.shape[:2]
        scale=min((tw-4)/pw, (th-4)/ph)
        new_w=max(1, int(pw*scale))
        new_h=max(1, int(ph*scale))
        patch_resized=cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        x_off=(tw-new_w)//2
        y_off=(th-new_h)//2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w]=patch_resized

        # baseline border (white)
        cv2.rectangle(canvas, (0, 0), (tw-1, th-1), (255, 255, 255), 1)
        if selected:
            # draw thick green border *inside* so tile size doesn't change
            cv2.rectangle(canvas, (2, 2), (tw-3, th-3), (0, 255, 0), 3)
        return canvas

    def _on_tile_click(self, t):
        obj_idx=self.tile_indices[t]
        if obj_idx is None:
            return
        if t in self.selected_tiles:
            self.selected_tiles.remove(t)
        else:
            self.selected_tiles.add(t)
        # redraw that tile only
        patch=self.objects[obj_idx]["patch"]
        selected=(t in self.selected_tiles)
        tile_img=self._make_tile_image(patch, selected)
        img_rgb=cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        photo=ImageTk.PhotoImage(pil_img)
        self.tile_photos[t]=photo
        self.tile_labels[t].config(image=photo)
        self._update_next_skip_button()

    def _update_next_skip_button(self):
        if self.selected_tiles:
            self.next_skip_btn.config(text="Next")
        else:
            self.next_skip_btn.config(text="Skip page")

    def _next_or_skip_page(self):
        #$ assign current label to selected tiles
        if self.selected_tiles and self.current_label != "trash":
            for t in self.selected_tiles:
                obj_idx=self.tile_indices[t]
                if obj_idx is not None:
                    self.labels[obj_idx]=self.current_label
        elif self.selected_tiles and self.current_label=="trash":
            for t in self.selected_tiles:
                obj_idx=self.tile_indices[t]
                if obj_idx is not None:
                    self.labels[obj_idx]="trash"

        self._update_label_counts_and_colors()
        #$ move to next page for this label
        self.current_page += 1
        self.selected_tiles.clear()
        self._update_grid()

    # ----- Finish -----

    def _finish(self):

        props_keys=["area", "avg_r", "avg_g", "avg_b", "aspect_ratio"]
        agg={}

        # Loop over all objects and their assigned labels
        for obj, lab in zip(self.objects, self.labels):
            # skip unlabeled and trash
            if lab is None or lab=="trash":
                continue

            props=obj["props"]

            if lab not in agg:
                agg[lab]={k: {"count": 0, "sum": 0.0, "sum_sq": 0.0} for k in props_keys}

            for k in props_keys:
                val=float(props[k])
                agg[lab][k]["count"] += 1
                agg[lab][k]["sum"] += val
                agg[lab][k]["sum_sq"] += val*val

        # Convert sums to mean/std
        label_stats={}
        for lab, pdict in agg.items():
            label_stats[lab]={}
            for k in props_keys:
                c=pdict[k]["count"]
                if c==0:
                    label_stats[lab][k]={"count": 0, "mean": 0.0, "std": 0.0}
                    continue
                s=pdict[k]["sum"]
                ss=pdict[k]["sum_sq"]
                mean=s/c
                var=max(0.0, ss/c-mean*mean)
                std=math.sqrt(var)
                label_stats[lab][k]={"count": c, "mean": mean, "std": std}

        # Load calib, store label_stats, save
        calib=load_calib()
        calib["label_stats"]=label_stats
        save_calib(calib)

        print(f"[Labeling] Saved label_stats for {len(label_stats)} labels into {CALIB_PATH}")
        self.top.destroy()



# -------------------------------------------------
# Calibration wizard
# -------------------------------------------------

class CalibrationApp:
    def __init__(self):
        self.screenshots_dir="calibration/screenshots"
        self.crops_dir="calibration/crops"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

        self.steps=[
            {"id": 1, "key": "hook_left",  "label": "Capture game screen (hook LEFT)",  "type": "capture"},
            {"id": 2, "key": "hook_right", "label": "Capture game screen (hook RIGHT)", "type": "capture"},
            {"id": 3, "key": "game_crop",  "label": "Select game screen crop",         "type": "crop"},
            {"id": 4, "key": "target",     "label": "Select target square",            "type": "target"},
            {"id": 5, "key": "hook_patch", "label": "Select hook patch",               "type": "hook_patch"},
            {"id": 6, "key": "merge_bg",   "label": "Merge LEFT/RIGHT hook backgrounds", "type": "merge"},
            {"id": 7, "key": "live_align", "label": "Live alignment (template match target)", "type": "align"},
            {"id": 8, "key": "levels",     "label": "Capture 7 levels with objects",   "type": "levels"},
            {"id": 9, "key": "tnt_threshold",
             "label": "Tune TNT background-subtraction threshold", "type": "tnt_threshold"},
            {"id": 10, "key": "label_objects",
             "label": "Label objects (captcha)",                  "type": "label"},
        ]
        self.step_status={s["id"]: "pending" for s in self.steps}
        self.current_step_idx=0
        self.step_status[self.steps[0]["id"]]="active"

        self.align_thread=None
        self.align_state={"stop": False}
        self.level_capture_count=0  # want 7

        self.root=tk.Tk()
        self.root.title("Calibration Wizard")

        self.title_font=("Segoe UI", 12, "bold")
        self.step_title_font=("Segoe UI", 10, "bold")
        self.text_font=("Segoe UI", 9)
        self.dot_font=("Segoe UI", 10, "bold")
        self.button_font=("Segoe UI", 10, "bold")

        main=tk.Frame(self.root, padx=8, pady=8)
        main.pack(fill="both", expand=True)

        tk.Label(main, text="Calibration Stages", font=self.title_font).pack(anchor="w")

        self.steps_frame=tk.Frame(main)
        self.steps_frame.pack(fill="x", pady=(4, 8))
        self.step_labels={}
        for s in self.steps:
            row=tk.Frame(self.steps_frame)
            row.pack(anchor="w")
            dot=tk.Label(row, text="●", font=self.dot_font, fg="#cc0000")
            dot.pack(side="left")
            lbl=tk.Label(row, text=f"Step {s['id']}: {s['label']}", font=self.text_font)
            lbl.pack(side="left", padx=4)
            self.step_labels[s["id"]]=(dot, lbl)

        tk.Frame(main, height=1, bg="#888").pack(fill="x", pady=4)

        self.current_step_title=tk.Label(main, text="", font=self.step_title_font)
        self.current_step_title.pack(anchor="w", pady=(4, 2))

        self.instructions_text=tk.StringVar(value="")
        self.instructions_label=tk.Label(
            main,
            textvariable=self.instructions_text,
            font=self.text_font,
            wraplength=360,
            justify="left",
        )
        self.instructions_label.pack(anchor="w", pady=(0, 6))

        btn_row=tk.Frame(main)
        btn_row.pack(fill="x", pady=(4, 0))

        self.btn_capture=tk.Button(
            btn_row,
            text="Do step",
            command=self.on_capture_click,
            font=self.button_font,
            bg="#0a6",
            fg="white",
        )
        self.btn_capture.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_skip=tk.Button(
            btn_row,
            text="Skip step",
            command=self.on_skip_step,
            font=self.button_font,
        )
        self.btn_skip.pack(side="left", padx=4)

        self.btn_restart=tk.Button(
            main,
            text="Restart from Step 1",
            command=self.on_restart,
            font=self.button_font,
        )
        self.btn_restart.pack(fill="x", pady=(6, 0))

        self.update_current_step_ui()
        self.position_bottom_right_inset()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def position_bottom_right_inset(self):
        self.root.update_idletasks()
        w=self.root.winfo_width()
        h=self.root.winfo_height()
        sw=self.root.winfo_screenwidth()
        sh=self.root.winfo_screenheight()

        target_x=int(sw*0.99)
        target_y=int(sh*0.9)

        x=target_x-w
        y=target_y-h
        x=max(0, x)
        y=max(0, y)

        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def refresh_step_status_ui(self):
        for s in self.steps:
            sid=s["id"]
            status=self.step_status[sid]
            dot, _=self.step_labels[sid]
            if status=="done":
                dot.config(fg="#00aa00")
            elif status=="active":
                dot.config(fg="#d4aa00")
            else:
                dot.config(fg="#cc0000")

    def update_current_step_ui(self):
        step=self.steps[self.current_step_idx]
        sid=step["id"]
        self.current_step_title.config(text=f"Current: Step {sid}")

        if sid in (1, 2):
            self.btn_capture.config(text="Capture screenshot")
        elif sid==3:
            self.btn_capture.config(text="Open cropping tool")
        elif sid==4:
            self.btn_capture.config(text="Open target tool")
        elif sid==5:
            self.btn_capture.config(text="Open hook patch tool")
        elif sid==6:
            self.btn_capture.config(text="Open merge background tool")
        elif sid==7:
            self.btn_capture.config(text="Confirm alignment")
            if self.align_thread is None:
                self.start_live_alignment()
        elif sid==8:
            self.btn_capture.config(text="Capture level")
        elif sid==9:
            self.btn_capture.config(text="Open TNT threshold tool")
        elif sid==10:
            self.btn_capture.config(text="Open labeling tool")
        else:
            self.btn_capture.config(text="Do step")

        if sid==1:
            instr="Step 1: Capture game screen with HOOK on LEFT."
        elif sid==2:
            instr="Step 2: Capture game screen with HOOK on RIGHT."
        elif sid==3:
            instr="Step 3: Open cropping tool and adjust game window edges."
        elif sid==4:
            instr="Step 4: Open target tool and adjust the target box."
        elif sid==5:
            instr="Step 5: Open hook patch tool and place the box over the idle hook."
        elif sid==6:
            instr="Step 6: Merge LEFT/RIGHT hook crops into a clean background."
        elif sid==7:
            instr="Step 7: Live align using target template.\nMove game window if needed, then confirm."
        elif sid==8:
            instr=f"Step 8: Capture 7 different level screens ({self.level_capture_count}/7)."
        elif sid==9:
            instr="Step 9: Select a screenshot with TNT, pick TNT box, and tune background-subtraction threshold."
        elif sid==10:
            instr="Step 10: Label detected objects using captcha-style grid."
        else:
            instr="Future step (unused)."

        self.instructions_text.set(instr)

        self.btn_capture.config(state="normal")
        self.btn_skip.config(state="normal")
        self.refresh_step_status_ui()

    def capture_fullscreen(self, step_id, key):
        with mss.mss() as sct:
            monitor=sct.monitors[0]
            grab=sct.grab(monitor)
            frame=np.array(grab)[:, :, :3]

        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"step{step_id:02d}_{key}_{ts}.png"
        path=os.path.join(self.screenshots_dir, filename)
        cv2.imwrite(path, frame)
        print(f"[Calibration] Saved screenshot for step {step_id} -> {path}")
        return path

    def run_crop_dialog(self):
        dlg=CropDialog(self.root, self.screenshots_dir, self.crops_dir)
        self.root.wait_window(dlg.top)

    def run_target_dialog(self):
        dlg=TargetDialog(self.root, self.crops_dir)
        self.root.wait_window(dlg.top)

    def run_hook_patch_dialog(self):
        dlg=HookPatchDialog(self.root, self.crops_dir)
        self.root.wait_window(dlg.top)

    def run_merge_background_dialog(self):
        dlg=MergeBackgroundDialog(self.root, self.crops_dir)
        self.root.wait_window(getattr(dlg, "top", self.root))

    def start_live_alignment(self):
        calib=load_calib()
        self.align_state={"stop": False}
        self.align_thread=threading.Thread(
            target=live_align_loop,
            args=(self.crops_dir, calib, self.align_state),
            daemon=True
        )
        self.align_thread.start()

    def run_threshold_tool(self):
        dlg=TNTThresholdDialog(self.root, self.screenshots_dir, self.crops_dir)
        self.root.wait_window(getattr(dlg, "top", self.root))

    def run_labeling_tool(self):
        calib=load_calib()
        all_paths=[
            p for p in glob.glob(os.path.join(self.screenshots_dir, "*.png"))
            if "levels" in os.path.basename(p)
        ]
        if not all_paths:
            print("[Labeling] No level screenshots for labeling.")
            return
        all_paths.sort(key=os.path.getmtime)
        level_paths=all_paths[-7:]  #$ use latest 7
        dlg=LabelingCaptchaDialog(self.root, level_paths, calib, self.crops_dir)
        self.root.wait_window(getattr(dlg, "top", self.root))

    def on_capture_click(self):
        step=self.steps[self.current_step_idx]
        sid=step["id"]

        try:
            if sid in (1, 2):
                self.capture_fullscreen(sid, step["key"])
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==3:
                self.run_crop_dialog()
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==4:
                self.run_target_dialog()
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==5:
                self.run_hook_patch_dialog()
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==6:
                self.run_merge_background_dialog()
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==7:
                if self.align_thread is not None:
                    self.align_state["stop"]=True
                    self.align_thread.join(timeout=1.0)
                    self.align_thread=None
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==8:
                if self.level_capture_count < 7:
                    idx=self.level_capture_count+1
                    self.capture_fullscreen(sid, f"levels_{idx}")
                    self.level_capture_count += 1
                    if self.level_capture_count >= 7:
                        self.step_status[sid]="done"
                        self.advance_step()
                    else:
                        self.update_current_step_ui()
                else:
                    self.step_status[sid]="done"
                    self.advance_step()

            elif sid==9:
                self.run_threshold_tool()
                self.step_status[sid]="done"
                self.advance_step()

            elif sid==10:
                self.run_labeling_tool()
                self.step_status[sid]="done"
                self.advance_step()

        except Exception as e:
            print(f"[Calibration] Error at step {sid}: {e}")

    def on_skip_step(self):
        step=self.steps[self.current_step_idx]
        sid=step["id"]
        print(f"[Calibration] Skipping step {sid}: {step['label']} (reusing data if present).")

        if sid==7 and self.align_thread is not None:
            self.align_state["stop"]=True
            self.align_thread.join(timeout=1.0)
            self.align_thread=None

        self.step_status[sid]="done"
        self.advance_step()

    def advance_step(self):
        if self.current_step_idx+1 < len(self.steps):
            self.current_step_idx += 1
            next_id=self.steps[self.current_step_idx]["id"]
            if self.step_status[next_id]=="pending":
                self.step_status[next_id]="active"
            self.update_current_step_ui()
        else:
            print("[Calibration] All steps complete.")
            self.root.destroy()

    def on_restart(self):
        print("[Calibration] Restarting from Step 1.")
        if self.align_thread is not None:
            self.align_state["stop"]=True
            self.align_thread.join(timeout=1.0)
            self.align_thread=None

        self.step_status={s["id"]: "pending" for s in self.steps}
        self.current_step_idx=0
        self.step_status[self.steps[0]["id"]]="active"
        self.level_capture_count=0
        self.update_current_step_ui()

    def on_close(self):
        if self.align_thread is not None:
            self.align_state["stop"]=True
            self.align_thread.join(timeout=1.0)
        self.root.destroy()


if __name__=="__main__":
    warmup_capture() #$ to prevent screen moving issue when capturing screen
    CalibrationApp()
