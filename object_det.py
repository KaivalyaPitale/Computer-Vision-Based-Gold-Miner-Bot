import os                         #$ Accessing folders/saving files
import json                       #$ Reading/Writing into JSON calibration file
import math                       #$ Calculations
import tkinter as tk              #$ GUI

import mss                        #$ Capturing screenshots
import numpy as np
import cv2
from PIL import Image, ImageTk    #$ For converting to GUI friendly image format

# -------------------------------------------------
# Paths/constants
# -------------------------------------------------

CALIB_PATH="calibration/game_calibration.json"
CROPS_DIR="calibration/crops"
TARGET_PATH=os.path.join(CROPS_DIR, "target.png")
BACKGROUND_PATH="background.png"

THRESHOLD_VALUE_FALLBACK=20
GROUND_LINE_Y=220
MIN_OBJECT_AREA=100

#$ Labels that we may classify as
VALID_LABELS=[
    "gold_small", "gold_medium", "gold_large", "gold_nugget",
    "rock", "stone", "bone", "skull", "tnt",
    "diamond", "mole", "mole_diamond", "bag",
    "mole_left", "mole_right",
    "mole_diamond_left", "mole_diamond_right",
]

#$ Value priorities (lower=more valuable)
VALUE_PRIORITY={
    "diamond": 0,
    "mole_diamond": 1,
    "mole_diamond_left": 1,
    "mole_diamond_right": 1,
    "bag": 2,
    "gold_large": 3,
    "gold_medium": 4,
    "gold_small": 5,
    "gold_nugget": 6,
    "skull": 7,
    "bone": 8,
    "rock": 9,
    "stone": 10,
    "mole": 11,
    "mole_left": 11,
    "mole_right": 11,
    "tnt": 12,
}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def warmup_capture(): #$ This function prevents GUI size from changing after taking screenshots
    _=mss.mss().monitors[0]


def load_calib():
    if not os.path.exists(CALIB_PATH):
        print("[Detect] Calibration file not found:", CALIB_PATH)
        return None
    with open(CALIB_PATH, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Classification helpers (mean+sigma)
# -------------------------------------------------

def classify_object(props, label_stats, max_sigma):

    if not label_stats:
        return None

    best_label=None
    best_score=None

    for lab, stats in label_stats.items():
        if lab not in VALID_LABELS:
            continue
        if stats["area"]["count"]==0:
            continue

        ok_count=0
        total_score=0.0
        valid=True

        for key, val in [
            ("area", props["area"]),
            ("avg_r", props["avg_r"]),
            ("avg_g", props["avg_g"]),
            ("avg_b", props["avg_b"]),
            ("aspect_ratio", props["aspect_ratio"]),
        ]:
            s=stats[key]
            if s["count"]==0:
                valid=False
                break

            mean=s["mean"]
            std=s["std"]
            diff=val-mean

            if std > 0:
                z=diff/std
                if abs(z) <= max_sigma:
                    ok_count += 1
                    total_score += z*z
            else:
                # zero variance: treat as match only if almost identical
                if abs(diff) <= 1e-6:
                    ok_count += 1

        if not valid:
            continue

        # require at least 4 out of 5 attributes to be within sigma
        if ok_count < 4:
            continue

        if best_score is None or total_score < best_score:
            best_score=total_score
            best_label=lab

    return best_label


def compute_debug_matches(props, label_stats, max_sigma, top_k=3):

    results=[]

    for lab, stats in label_stats.items():
        if stats["area"]["count"]==0:
            continue

        total_score=0.0
        any_valid=True
        max_abs_z=0.0

        for key, val in [
            ("area", props["area"]),
            ("avg_r", props["avg_r"]),
            ("avg_g", props["avg_g"]),
            ("avg_b", props["avg_b"]),
            ("aspect_ratio", props["aspect_ratio"]),
        ]:
            s=stats[key]
            if s["count"]==0:
                any_valid=False
                break
            mean=s["mean"]
            std=s["std"]
            diff=val-mean
            if std > 0:
                z=diff/std
                total_score += z*z
                max_abs_z=max(max_abs_z, abs(z))
            else:
                if abs(diff) > 1e-6:
                    total_score += 1e6
                    max_abs_z=max(max_abs_z, 1e3)
                else:
                    max_abs_z=max(max_abs_z, 0.0)

        if not any_valid:
            continue

        results.append({
            "label": lab,
            "score": total_score,
            "max_abs_z": max_abs_z,
        })

    results.sort(key=lambda r: r["score"])
    return results[:top_k]


# -------------------------------------------------
# Detector core
# -------------------------------------------------

class Detector:
    def __init__(self):
        self.calib=load_calib()
        self.ready=False

        if self.calib is None:
            print("[Detect] No calibration loaded; cannot run detection.")
            return

        required_keys=["window_size", "target_center", "target_size", "label_stats"]
        missing=[k for k in required_keys if k not in self.calib]
        if missing:
            print("[Detect] Missing calibration keys:", missing)
            return

        if not os.path.exists(BACKGROUND_PATH):
            print("[Detect] Missing background image:", BACKGROUND_PATH)
            return

        self.bg=cv2.imread(BACKGROUND_PATH)
        if self.bg is None:
            print("[Detect] Failed to load background image:", BACKGROUND_PATH)
            return

        if not os.path.exists(TARGET_PATH):
            print("[Detect] Missing target template:", TARGET_PATH)
            return

        self.target_img=cv2.imread(TARGET_PATH)
        if self.target_img is None:
            print("[Detect] Failed to load target template:", TARGET_PATH)
            return

        self.win_w, self.win_h=self.calib["window_size"]
        self.cx, self.cy=self.calib["target_center"]
        self.target_size=self.calib["target_size"]
        self.label_stats=self.calib.get("label_stats", {})
        self.threshold_value=int(self.calib.get("threshold_value", THRESHOLD_VALUE_FALLBACK))

        # hook center (for polar coordinates)
        hook_center=self.calib.get("hook_center")
        if isinstance(hook_center, (list, tuple)) and len(hook_center)==2:
            self.hook_cx=int(hook_center[0])
            self.hook_cy=int(hook_center[1])
            print(f"[Detect] hook_center: ({self.hook_cx}, {self.hook_cy})")
        else:
            self.hook_cx=None
            self.hook_cy=None
            print("[Detect] No hook_center in calibration; polar coords will be skipped.")

        #$ hook patch params (for hook angle)
        hook_patch_center=self.calib.get("hook_patch_center")
        self.hook_patch_w=int(self.calib.get("hook_patch_width", 0))
        self.hook_patch_h=int(self.calib.get("hook_patch_height", 0))
        if isinstance(hook_patch_center, (list, tuple)) and len(hook_patch_center)==2:
            self.hook_patch_cx=int(hook_patch_center[0])
            self.hook_patch_cy=int(hook_patch_center[1])
            print(
                f"[Detect] hook_patch center=({self.hook_patch_cx},{self.hook_patch_cy}) "
                f"size=({self.hook_patch_w}x{self.hook_patch_h})"
            )
        else:
            self.hook_patch_cx=None
            self.hook_patch_cy=None
            print("[Detect] No hook_patch_* in calibration; hook angle will be skipped.")

        half=self.target_size//2
        target_box_tl_x=self.cx-half
        target_box_tl_y=self.cy-half
        self.offset_x=-target_box_tl_x
        self.offset_y=-target_box_tl_y

        #$ Preresize background to game window size
        self.bg_resized=cv2.resize(self.bg, (self.win_w, self.win_h))

        self.ready=True
        print("[Detect] Detector initialized and ready.")
        print(f"[Detect] Using threshold_value={self.threshold_value}")
        print(f"[Detect] GROUND_LINE_Y={GROUND_LINE_Y} (pixels from top)")

    def capture_screen(self):
        with mss.mss() as sct:
            monitor=sct.monitors[0]
            grab=sct.grab(monitor)
            frame=np.array(grab)[:, :, :3]
        return frame

    def align_game_window(self, frame):
        # template match with target image
        res=cv2.matchTemplate(frame, self.target_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc=cv2.minMaxLoc(res)
        print(f"[Detect] Template match score: {max_val:.3f}")

        if max_val < 0.6:
            print("[Detect] Target not confidently found on screen.")
            return None

        tx, ty=max_loc
        gx=tx+self.offset_x
        gy=ty+self.offset_y

        fh, fw=frame.shape[:2]
        gx=max(0, min(gx, fw-self.win_w))
        gy=max(0, min(gy, fh-self.win_h))

        game=frame[gy:gy+self.win_h, gx:gx+self.win_w].copy()
        if game.shape[1] != self.win_w or game.shape[0] != self.win_h:
            game=cv2.resize(game, (self.win_w, self.win_h))

        return game

    # ---------- Hook patch helpers ----------

    def get_hook_patch_rect(self, game_img):
        if self.hook_cx is None or self.hook_cy is None:
            return None
        if self.hook_patch_cx is None or self.hook_patch_cy is None:
            return None
        if self.hook_patch_w <= 0 or self.hook_patch_h <= 0:
            return None

        h_img, w_img=game_img.shape[:2]

        hx, hy=self.hook_cx, self.hook_cy
        pcx, pcy=self.hook_patch_cx, self.hook_patch_cy
        pw, ph=self.hook_patch_w, self.hook_patch_h

        # Full hook patch rect
        x1=int(pcx-pw//2)
        x2=int(pcx+pw//2)
        y_top=int(pcy-ph//2)
        y_bottom=int(pcy+ph//2)

        # Use only region below hook center
        y1=max(hy, y_top)
        y2=y_bottom

        # Clamp to image bounds
        x1=max(0, min(x1, w_img-1))
        x2=max(0, min(x2, w_img))
        y1=max(0, min(y1, h_img-1))
        y2=max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def compute_hook_angle(self, game_img):
        rect=self.get_hook_patch_rect(game_img)
        if rect is None:
            return None

        x1, y1, x2, y2=rect

        roi_game=game_img[y1:y2, x1:x2]
        roi_bg=self.bg_resized[y1:y2, x1:x2]

        diff=cv2.absdiff(roi_game, roi_bg)
        gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary=cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        ys, xs=np.nonzero(binary)
        if xs.size==0:
            return None

        cx_roi=xs.mean()
        cy_roi=ys.mean()

        cx=x1+cx_roi
        cy=y1+cy_roi

        dx=cx-self.hook_cx
        dy=cy-self.hook_cy

        # angle: 0° is straight down,+right,-left
        theta=math.atan2(dx, dy)
        theta_deg=math.degrees(theta)
        return theta_deg

    # ---------- Obstruction helpers ----------

    @staticmethod
    def _add_segment(segs, new_seg):
        a, b=new_seg
        if b <= a:
            return segs[:]
        all_segs=segs[:]+[(a, b)]
        all_segs.sort(key=lambda s: s[0])
        merged=[]
        for s, e in all_segs:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1]=max(merged[-1][1], e)
        return [(s, e) for s, e in merged]

    @staticmethod
    def _intersect_segments(segs, obj_range):
        a0, a1=obj_range
        if a1 <= a0:
            return []
        inter=[]
        for s, e in segs:
            ss=max(a0, s)
            ee=min(a1, e)
            if ee > ss:
                inter.append((ss, ee))
        if not inter:
            return []
        inter.sort(key=lambda x: x[0])
        merged=[[inter[0][0], inter[0][1]]]
        for s, e in inter[1:]:
            if s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1]=max(merged[-1][1], e)
        return [(s, e) for s, e in merged]

    @staticmethod
    def _compute_unblocked_segments(angle_range, blocked_segments):
        a0, a1=angle_range
        inter=Detector._intersect_segments(blocked_segments, (a0, a1))
        if not inter:
            return [(a0, a1)]
        unblocked=[]
        cur=a0
        for s, e in inter:
            if s > cur:
                unblocked.append((cur, s))
            cur=max(cur, e)
        if cur < a1:
            unblocked.append((cur, a1))
        return unblocked

    def _compute_obstructions(self, detections):
        idxs=[]

        # 1) Extract angle ranges+distance
        for i, det in enumerate(detections):
            polar=det.get("polar")
            if not polar or not polar.get("points"):
                continue
            pts=polar["points"]
            if len(pts) < 2:
                continue

            angles=[p["theta_deg"] for p in pts]
            radii=[p["r"] for p in pts]

            a0=min(angles)
            a1=max(angles)
            if a1 <= a0:
                continue

            r_near=min(radii)

            det["angle_range"]=(a0, a1)
            det["r_near"]=r_near
            det["blocked_segments"]=[]
            det["obstructed"]=False
            idxs.append(i)

        if not idxs:
            return

        # 2) Sort by distance (near -> far)
        idxs_sorted=sorted(idxs, key=lambda i: detections[i]["r_near"])

        # 3) Propagate blocking ranges
        for pos_i, i in enumerate(idxs_sorted):
            det_i=detections[i]
            label_i=det_i.get("label")
            if not label_i:
                # Unknown object does not block others
                continue

            a_i0, a_i1=det_i["angle_range"]
            seg_i=(a_i0, a_i1)
            r_i=det_i["r_near"]

            # Block all farther objects with polar info
            for j in idxs_sorted[pos_i+1:]:
                det_j=detections[j]
                r_j=det_j["r_near"]
                if r_j <= r_i+1e-6:
                    continue
                det_j["blocked_segments"]=self._add_segment(
                    det_j["blocked_segments"], seg_i
                )

        # 4) Decide obstructed/not for each object
        for i in idxs:
            det=detections[i]
            a0, a1=det["angle_range"]
            segs=det.get("blocked_segments", [])
            inter=self._intersect_segments(segs, (a0, a1))
            span_obj=a1-a0
            span_blocked=sum(e-s for (s, e) in inter)
            if span_obj <= 0:
                det["obstructed"]=False
            else:
                if span_blocked >= span_obj-1e-3:
                    det["obstructed"]=True
                else:
                    det["obstructed"]=False

    # ---------- Object selection helpers ----------

    def select_best_object(self, detections):
        best_idx=None
        best_priority=None
        best_r=None

        for i, det in enumerate(detections):
            label=det.get("label")
            if not label:
                continue

            # must have angular info & be unobstructed
            if det.get("obstructed") is not False:
                continue
            if "angle_range" not in det or "r_near" not in det:
                continue

            pri=VALUE_PRIORITY.get(label)
            if pri is None:
                continue

            r=det["r_near"]
            if best_idx is None:
                best_idx=i
                best_priority=pri
                best_r=r
            else:
                if pri < best_priority or (pri==best_priority and r < best_r):
                    best_idx=i
                    best_priority=pri
                    best_r=r

        return best_idx

    def decide_deploy(self, det, hook_angle_deg):
        if det is None or hook_angle_deg is None:
            return False
        if "angle_range" not in det:
            return False

        a0, a1=det["angle_range"]
        blocked=det.get("blocked_segments", [])
        unblocked=self._compute_unblocked_segments((a0, a1), blocked)
        angle=hook_angle_deg

        # simple check: angle inside any [u0,u1]
        for u0, u1 in unblocked:
            if u0-1e-3 <= angle <= u1+1e-3:
                return True
        return False

    # ---------- Object detection ----------

    def detect_objects(self, game_img, max_sigma):
        hook_angle_deg=self.compute_hook_angle(game_img)

        diff=cv2.absdiff(game_img, self.bg_resized)
        gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary=cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        if GROUND_LINE_Y < binary.shape[0]:
            binary[:GROUND_LINE_Y, :]=0

        contours, _=cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections=[]
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area < MIN_OBJECT_AREA:
                continue

            x, y, w, h=cv2.boundingRect(cnt)

            mask=np.zeros(game_img.shape[:2], np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            roi=game_img[y:y+h, x:x+w]
            mask_roi=mask[y:y+h, x:x+w]
            mean_bgr=cv2.mean(roi, mask=mask_roi)[:3]

            props={
                "area": float(area),
                "avg_b": mean_bgr[0],
                "avg_g": mean_bgr[1],
                "avg_r": mean_bgr[2],
                "aspect_ratio": (w/h if h > 0 else 0.0),
            }

            label=classify_object(props, self.label_stats, max_sigma)
            bbox=(x, y, w, h)

            # ------------------------
            # Polar coordinates logic
            # ------------------------
            polar_info=None
            if self.hook_cx is not None and self.hook_cy is not None:
                hx=self.hook_cx
                hy=self.hook_cy

                left=x
                right=x+w
                top=y
                bottom=y+h

                if right <= hx:
                    # Entirely left
                    relation="left"
                    points_of_interest=[
                        ("top_left", (left, top)),
                        ("bottom_right", (right, bottom)),
                    ]
                elif left >= hx:
                    # Entirely right
                    relation="right"
                    points_of_interest=[
                        ("top_right", (right, top)),
                        ("bottom_left", (left, bottom)),
                    ]
                else:
                    # Crosses hook x
                    relation="cross"
                    points_of_interest=[
                        ("top_left", (left, top)),
                        ("top_right", (right, top)),
                    ]

                polar_points=[]
                for corner_name, (px, py) in points_of_interest:
                    dx=px-hx
                    dy=py-hy
                    r=math.hypot(dx, dy)
                    # angle from vertical downward (0 deg=straight down),
                    # positive to the right, negative to the left
                    theta=math.atan2(dx, dy)
                    theta_deg=math.degrees(theta)

                    polar_points.append({
                        "corner": corner_name,
                        "point": (px, py),
                        "dx": dx,
                        "dy": dy,
                        "r": r,
                        "theta_rad": theta,
                        "theta_deg": theta_deg,
                    })

                polar_info={
                    "hook_center": (hx, hy),
                    "relation": relation,
                    "points": polar_points,
                }

            detections.append({
                "bbox": bbox,
                "props": props,
                "label": label,
                "polar": polar_info,  # can be None if hook_center missing
            })

        # compute obstruction state for each object
        self._compute_obstructions(detections)

        return detections, diff, binary, hook_angle_deg

    def draw_detections_full(
        self,
        game_img,
        detections,
        show_labels=True,
        show_angles=True,
        show_radius=True,
        hook_angle_deg=None,
        best_index=None,
        deploy_decision=None,
    ):

        out=game_img.copy()
        h_img, w_img=out.shape[:2]

        for idx, det in enumerate(detections):
            x, y, w, h=det["bbox"]
            label=det["label"] if det["label"] else "?"

            # bbox color based on obstruction/best object
            if best_index is not None and idx==best_index:
                color=(255, 0, 255)      # pink for object-of-interest
            else:
                obstructed=det.get("obstructed", None)
                if obstructed is True:
                    color=(0, 0, 0)      # fully obstructed
                elif obstructed is False and "obstructed" in det:
                    color=(0, 255, 0)    # unobstructed
                else:
                    color=(128, 128, 128)  # no polar info

            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

            # main label above the box (toggled)
            if show_labels:
                cv2.putText(out, label, (x, max(0, y-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            polar=det.get("polar")
            if polar and polar.get("points"):
                # corner angles
                if show_angles:
                    for p in polar["points"]:
                        px, py=p["point"]
                        angle_text=f"{p['theta_deg']:.1f}"

                        tx=int(px)
                        ty=int(py)

                        tx=max(0, min(tx, w_img-1))
                        ty=max(10, min(ty, h_img-1))

                        cv2.putText(
                            out,
                            angle_text,
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),  # white
                            1
                        )

                # radius at center using min of the two radii
                if show_radius:
                    rs=[p["r"] for p in polar["points"] if "r" in p]
                    if rs:
                        r_val=min(rs)
                        r_text=f"{r_val:.1f}"

                        cx=x+w//2
                        cy=y+h//2

                        cx=max(0, min(cx, w_img-1))
                        cy=max(10, min(cy, h_img-1))

                        cv2.putText(
                            out,
                            r_text,
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 0, 255),  # pink
                            1
                        )

        # Hook angle text+scope line
        if hook_angle_deg is not None and self.hook_cx is not None and self.hook_cy is not None:
            hx=max(0, min(self.hook_cx, w_img-1))
            hy=max(0, min(self.hook_cy, h_img-1))

            # Draw hook angle text (always)
            text=f"{hook_angle_deg:.1f}"
            ty=max(10, hy-5)
            cv2.putText(
                out,
                text,
                (hx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),  # yellow-ish
                1
            )

            # Draw scope line from hook center to edge of screen
            rad=math.radians(hook_angle_deg)
            dx=math.sin(rad)
            dy=math.cos(rad)

            candidates=[]

            # bottom edge
            if dy > 1e-6:
                t_bottom=(h_img-1-hy)/dy
                candidates.append(t_bottom)
            # top edge
            if dy < -1e-6:
                t_top=(0-hy)/dy
                candidates.append(t_top)
            # right edge
            if dx > 1e-6:
                t_right=(w_img-1-hx)/dx
                candidates.append(t_right)
            # left edge
            if dx < -1e-6:
                t_left=(0-hx)/dx
                candidates.append(t_left)

            t_candidates=[t for t in candidates if t > 0]
            if t_candidates:
                t_end=min(t_candidates)
                ex=int(round(hx+dx*t_end))
                ey=int(round(hy+dy*t_end))
                ex=max(0, min(ex, w_img-1))
                ey=max(0, min(ey, h_img-1))

                cv2.line(out, (hx, hy), (ex, ey), (255, 255, 0), 1)

        # Deploy Hook? Yes/No
        if deploy_decision is not None and self.hook_cx is not None and self.hook_cy is not None:
            hx=max(0, min(self.hook_cx, w_img-1))
            hy=max(0, min(self.hook_cy, h_img-1))

            midx=hx//2
            ytext=max(10, min(hy, h_img-10))

            base_text="Deploy Hook? "
            ans_text="Yes" if deploy_decision else "No"

            font=cv2.FONT_HERSHEY_SIMPLEX
            scale=0.6
            thickness=1

            size_base, _=cv2.getTextSize(base_text, font, scale, thickness)

            # "Deploy Hook?" in white
            cv2.putText(
                out,
                base_text,
                (midx, ytext),
                font,
                scale,
                (255, 255, 255),
                thickness
            )

            # "Yes"/"No" in green/red
            x_ans=midx+size_base[0]
            color_ans=(0, 255, 0) if deploy_decision else (0, 0, 255)
            cv2.putText(
                out,
                ans_text,
                (x_ans, ytext),
                font,
                scale,
                color_ans,
                thickness
            )

        return out


# -------------------------------------------------
# Detection GUI
# -------------------------------------------------

class DetectApp:
    def __init__(self):
        warmup_capture()
        self.detector=Detector()

        self.root=tk.Tk()
        self.root.title("GoldMiner Detector")

        frame=tk.Frame(self.root, padx=10, pady=10)
        frame.pack()

        self.status_var=tk.StringVar()
        if self.detector.ready:
            self.status_var.set("Ready. Press 'Capture & Detect'.")
        else:
            self.status_var.set("NOT READY. Check console for errors.")

        tk.Label(frame, textvariable=self.status_var, justify="left").pack(pady=(0, 8))

        # Sigma slider (now up to 10)
        slider_frame=tk.Frame(frame)
        slider_frame.pack(pady=(0, 6), fill="x")
        tk.Label(slider_frame, text="Label σ threshold:").pack(anchor="w")
        self.sigma_var=tk.DoubleVar(value=4.0)
        self.sigma_scale=tk.Scale(
            slider_frame,
            from_=1.0,
            to=10.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.sigma_var,
            length=220
        )
        self.sigma_scale.pack(anchor="w")

        # Debug compare label dropdown
        dbg_frame=tk.Frame(frame)
        dbg_frame.pack(pady=(0, 6), fill="x")
        tk.Label(dbg_frame, text="Debug compare label:").pack(anchor="w")

        if self.detector.ready:
            labels_for_debug=sorted(
                [lab for lab in self.detector.label_stats.keys() if lab in VALID_LABELS]
            )
        else:
            labels_for_debug=[]

        self.debug_compare_label_var=tk.StringVar(value="(auto)")
        debug_options=["(auto)"]+labels_for_debug
        self.debug_label_menu=tk.OptionMenu(dbg_frame, self.debug_compare_label_var, *debug_options)
        self.debug_label_menu.pack(anchor="w")

        # Overlay options: labels/angle/distance
        overlay_frame=tk.Frame(frame)
        overlay_frame.pack(pady=(0, 6), fill="x")
        tk.Label(overlay_frame, text="Show overlays:").pack(anchor="w")
        opts_row=tk.Frame(overlay_frame)
        opts_row.pack(anchor="w")

        self.show_label_var=tk.BooleanVar(value=True)
        self.show_angle_var=tk.BooleanVar(value=True)
        self.show_radius_var=tk.BooleanVar(value=True)

        tk.Checkbutton(opts_row, text="labels", variable=self.show_label_var)\
            .pack(side="left", padx=4)
        tk.Checkbutton(opts_row, text="angle", variable=self.show_angle_var)\
            .pack(side="left", padx=4)
        tk.Checkbutton(opts_row, text="distance", variable=self.show_radius_var)\
            .pack(side="left", padx=4)

        self.btn_capture=tk.Button(
            frame,
            text="Capture & Detect",
            width=25,
            command=self.on_capture_click,
            state=("normal" if self.detector.ready else "disabled")
        )
        self.btn_capture.pack(pady=4)

        tk.Button(frame, text="Quit", width=25, command=self.on_quit).pack(pady=4)

        # result window state
        self.result_win=None
        self.result_canvas=None
        self.result_photo=None

        self.last_game=None
        self.last_detections=None
        self.base_full_bgr=None
        self.display_scale=1.0

        # for background subtraction+hook patch debug
        self.last_diff=None
        self.last_binary=None
        self.bg_button_rect=None  # (x1,y1,x2,y2) in original full_bgr coords
        self.last_hook_angle=None

        self.position_bottom_right_inset()

        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
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

    def _draw_background_button(self, img):
        h, w=img.shape[:2]
        btn_w=170
        btn_h=26
        x1=5
        y1=h-btn_h-5
        x2=x1+btn_w
        y2=y1+btn_h

        # filled black box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(img, "background subtraction", (x1+5, y1+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        self.bg_button_rect=(x1, y1, x2, y2)

    def update_result_window(self, full_bgr, set_as_base=False):
        if set_as_base:
            self.base_full_bgr=full_bgr.copy()

        # always redraw background button on the full image
        self._draw_background_button(full_bgr)

        h, w=full_bgr.shape[:2]
        max_display_width=900
        if w > max_display_width:
            scale=max_display_width/w
            disp_w=int(w*scale)
            disp_h=int(h*scale)
            display_bgr=cv2.resize(full_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            scale=1.0
            display_bgr=full_bgr.copy()

        self.display_scale=scale

        img_rgb=cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        self.result_photo=ImageTk.PhotoImage(pil_img)

        if self.result_win is None or not self.result_win.winfo_exists():
            self.result_win=tk.Toplevel(self.root)
            self.result_win.title("Detected Objects")
            self.result_canvas=tk.Label(self.result_win)
            self.result_canvas.pack()
            self.result_canvas.bind("<Button-1>", self.on_result_click)

        self.result_canvas.config(image=self.result_photo)

    def _click_in_bg_button(self, gx, gy):
        if self.bg_button_rect is None:
            return False
        x1, y1, x2, y2=self.bg_button_rect
        return (x1 <= gx < x2) and (y1 <= gy < y2)

    def _click_in_hook_patch(self, gx, gy):
        if self.last_game is None:
            return False
        rect=self.detector.get_hook_patch_rect(self.last_game)
        if rect is None:
            return False
        x1, y1, x2, y2=rect
        return (x1 <= gx < x2) and (y1 <= gy < y2)

    def show_background_subtraction_window(self):
        if self.last_game is None or self.detector.bg_resized is None:
            print("[Debug] No game/background available for subtraction view.")
            return

        game=self.last_game
        bg=self.detector.bg_resized
        t=self.detector.threshold_value

        diff=cv2.absdiff(game, bg)
        gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary=cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        # 3-panel view: game, bg, binary
        panels=[
            ("Game", game),
            ("Background", bg),
            ("Binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
        ]

        tile_w=220
        tile_h=220
        margin=10

        total_w=3*tile_w+4*margin
        total_h=tile_h+2*margin+30

        view=np.zeros((total_h, total_w, 3), dtype=np.uint8)
        view[:]=(0, 0, 0)

        for i, (name, img) in enumerate(panels):
            ih, iw=img.shape[:2]
            scale=min(tile_w/iw, tile_h/ih)
            new_w=max(1, int(iw*scale))
            new_h=max(1, int(ih*scale))
            resized=cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            x_off=margin+i*(tile_w+margin)+(tile_w-new_w)//2
            y_off=margin+(tile_h-new_h)//2
            view[y_off:y_off+new_h, x_off:x_off+new_w]=resized
            cv2.rectangle(view, (x_off, y_off),
                          (x_off+tile_w-1, y_off+tile_h-1), (255, 255, 255), 1)
            cv2.putText(view, name, (x_off+5, y_off+tile_h+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # show using Tk
        win=tk.Toplevel(self.root)
        win.title("Background subtraction")
        img_rgb=cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        photo=ImageTk.PhotoImage(pil_img)
        lbl=tk.Label(win, image=photo)
        lbl.image=photo
        lbl.pack()

    def show_hook_patch_window(self):
        if self.last_game is None or self.detector.bg_resized is None:
            print("[Debug] No game/background for hook patch view.")
            return

        rect=self.detector.get_hook_patch_rect(self.last_game)
        if rect is None:
            print("[Debug] No valid hook patch rect.")
            return

        x1, y1, x2, y2=rect
        game_roi=self.last_game[y1:y2, x1:x2]
        bg_roi=self.detector.bg_resized[y1:y2, x1:x2]

        t=self.detector.threshold_value
        diff=cv2.absdiff(game_roi, bg_roi)
        gray=cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary=cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        panels=[
            ("Hook ROI", game_roi),
            ("Hook BG", bg_roi),
            ("Hook mask", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
        ]

        tile_w=220
        tile_h=220
        margin=10

        total_w=3*tile_w+4*margin
        total_h=tile_h+2*margin+30

        view=np.zeros((total_h, total_w, 3), dtype=np.uint8)
        view[:]=(0, 0, 0)

        for i, (name, img) in enumerate(panels):
            ih, iw=img.shape[:2]
            scale=min(tile_w/iw, tile_h/ih)
            new_w=max(1, int(iw*scale))
            new_h=max(1, int(ih*scale))
            resized=cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            x_off=margin+i*(tile_w+margin)+(tile_w-new_w)//2
            y_off=margin+(tile_h-new_h)//2
            view[y_off:y_off+new_h, x_off:x_off+new_w]=resized
            cv2.rectangle(view, (x_off, y_off),
                          (x_off+tile_w-1, y_off+tile_h-1), (255, 255, 255), 1)
            cv2.putText(view, name, (x_off+5, y_off+tile_h+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        win=tk.Toplevel(self.root)
        win.title("Hook patch subtraction/mask")
        img_rgb=cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb)
        photo=ImageTk.PhotoImage(pil_img)
        lbl=tk.Label(win, image=photo)
        lbl.image=photo
        lbl.pack()

    def _compute_label_zscores(self, props, label, max_sigma):
        if not self.detector or not self.detector.label_stats:
            return None

        stats_all=self.detector.label_stats
        if label not in stats_all:
            return None

        stat=stats_all[label]
        if stat["area"]["count"]==0:
            return None

        keys_and_vals=[
            ("area", props["area"]),
            ("avg_r", props["avg_r"]),
            ("avg_g", props["avg_g"]),
            ("avg_b", props["avg_b"]),
            ("aspect_ratio", props["aspect_ratio"]),
        ]
        results=[]

        for key, val in keys_and_vals:
            s=stat[key]
            mean=s["mean"]
            std=s["std"]
            if s["count"]==0:
                z=None
                inside=False
            else:
                diff=val-mean
                if std > 0:
                    z=diff/std
                    inside=abs(z) <= max_sigma
                else:
                    # zero variance
                    if abs(diff) <= 1e-6:
                        z=0.0
                        inside=True
                    else:
                        z=float("inf")
                        inside=False

            results.append({
                "key": key,
                "value": val,
                "mean": mean,
                "std": std,
                "z": z,
                "inside": inside,
            })

        return results

    def on_result_click(self, event):
        if self.last_game is None or self.last_detections is None or self.base_full_bgr is None:
            return

        # scale back to original coords
        scale=self.display_scale if self.display_scale > 0 else 1.0
        gx=int(event.x/scale)
        gy=int(event.y/scale)

        # Click priority: bg button -> hook patch -> object debug
        if self._click_in_bg_button(gx, gy):
            self.show_background_subtraction_window()
            return

        if self._click_in_hook_patch(gx, gy):
            self.show_hook_patch_window()
            return

        chosen=None
        for det in self.last_detections:
            x, y, w, h=det["bbox"]
            if x <= gx < x+w and y <= gy < y+h:
                chosen=det
                break

        debug_bgr=self.base_full_bgr.copy()

        if chosen is None:
            print("[Debug] No contour/object under click.")
            self.update_result_window(debug_bgr, set_as_base=False)
            return

        props=chosen["props"]
        max_sigma=self.sigma_var.get()
        matches=compute_debug_matches(props, self.detector.label_stats, max_sigma, top_k=3)

        # Build text lines
        lines_meta=[]
        lines_meta.append(f"area={props['area']:.1f}  ar={props['aspect_ratio']:.3f}")
        lines_meta.append(f"R={props['avg_r']:.1f}  G={props['avg_g']:.1f}  B={props['avg_b']:.1f}")

        # Debug label choice
        selected_label=self.debug_compare_label_var.get()
        show_zscores=None
        if selected_label != "(auto)":
            show_zscores=self._compute_label_zscores(props, selected_label, max_sigma)

        z_lines=[]  # list of (text, color)
        if selected_label != "(auto)":
            if show_zscores is None:
                z_lines.append((f"Compare vs {selected_label}: no stats", (0, 0, 255)))
            else:
                z_lines.append(
                    (f"Compare vs {selected_label} (σ={max_sigma:.1f})", (255, 255, 255))
                )
                pretty_name={
                    "area": "Area",
                    "avg_r": "R",
                    "avg_g": "G",
                    "avg_b": "B",
                    "aspect_ratio": "Aspect",
                }
                for r in show_zscores:
                    key=r["key"]
                    val=r["value"]
                    mean=r["mean"]
                    std=r["std"]
                    z=r["z"]
                    inside=r["inside"]

                    if z is None:
                        txt=f"{pretty_name.get(key, key)}: v={val:.2f}  mean={mean:.2f}  (no std)"
                        color=(0, 0, 255)
                    else:
                        if math.isinf(z):
                            ztxt="inf"
                        else:
                            ztxt=f"{z:.2f}"
                        txt=(
                            f"{pretty_name.get(key, key)}: v={val:.2f}  "
                            f"mean={mean:.2f}  std={std:.2f}  z={ztxt}"
                        )
                        color=(0, 255, 0) if inside else (0, 0, 255)
                    z_lines.append((txt, color))

        # Top matches lines (always white)
        match_lines=[]
        for i, m in enumerate(matches, start=1):
            inside=f"(<= {max_sigma:.1f}σ)" if m["max_abs_z"] <= max_sigma else f"(> {max_sigma:.1f}σ)"
            match_lines.append(
                (f"{i}) {m['label']}  score={m['score']:.2f} {inside}", (255, 255, 255))
            )

        # Prepare overlay box
        text_x, text_y=5, 15
        line_height=16

        total_lines=len(lines_meta)+len(z_lines)+len(match_lines)
        box_height=line_height*total_lines+8
        box_width=420

        overlay=debug_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
        alpha=0.5
        debug_bgr[0:box_height, 0:box_width]=cv2.addWeighted(
            overlay[0:box_height, 0:box_width], alpha,
            debug_bgr[0:box_height, 0:box_width], 1-alpha,
            0.0
        )

        # Draw meta lines (always white)
        yy=text_y
        for text in lines_meta:
            cv2.putText(debug_bgr, text, (text_x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            yy += line_height

        # Draw z-score lines (color per-line)
        for text, color in z_lines:
            cv2.putText(debug_bgr, text, (text_x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            yy += line_height

        # Draw match lines
        for text, color in match_lines:
            cv2.putText(debug_bgr, text, (text_x, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            yy += line_height

        self.update_result_window(debug_bgr, set_as_base=False)

    def on_capture_click(self):
        if not self.detector.ready:
            self.status_var.set("Detector not ready.")
            return

        self.status_var.set("Capturing and detecting...")
        self.root.update_idletasks()

        try:
            frame=self.detector.capture_screen()
            game=self.detector.align_game_window(frame)
            if game is None:
                self.status_var.set("Target not found. Is the game visible?")
                return

            max_sigma=self.sigma_var.get()
            detections, diff, binary, hook_angle_deg=self.detector.detect_objects(game, max_sigma)
            n_labeled=sum(1 for d in detections if d["label"])
            print(f"[Detect] Found {len(detections)} objects, {n_labeled} labeled (σ={max_sigma:.1f}).")

            # Select best object and decide deploy
            best_index=self.detector.select_best_object(detections)
            if best_index is not None:
                det_best=detections[best_index]
            else:
                det_best=None

            if det_best is not None and hook_angle_deg is not None:
                deploy=self.detector.decide_deploy(det_best, hook_angle_deg)
            else:
                deploy=False

            self.last_game=game
            self.last_detections=detections
            self.last_diff=diff
            self.last_binary=binary
            self.last_hook_angle=hook_angle_deg

            full_bgr=self.detector.draw_detections_full(
                game,
                detections,
                show_labels=self.show_label_var.get(),
                show_angles=self.show_angle_var.get(),
                show_radius=self.show_radius_var.get(),
                hook_angle_deg=hook_angle_deg,
                best_index=best_index,
                deploy_decision=deploy,
            )

            self.update_result_window(full_bgr, set_as_base=True)

            hook_msg=f", hook={hook_angle_deg:.1f}°" if hook_angle_deg is not None else ""
            deploy_msg="Yes" if deploy else "No"
            self.status_var.set(
                f"Done. Objects: {len(detections)}, labeled: {n_labeled}. "
                f"σ={max_sigma:.1f} (threshold={self.detector.threshold_value})"
                f"{hook_msg}  Deploy Hook? {deploy_msg}"
            )
        except Exception as e:
            print("[Detect] Error during detection:", e)
            self.status_var.set(f"Error: {e}")

    def on_quit(self):
        self.root.destroy()


if __name__=="__main__":
    DetectApp()
