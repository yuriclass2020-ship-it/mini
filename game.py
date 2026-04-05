"""
벚꽃 잎 잡기 게임 (Hand Gesture Cherry Blossom Catcher)

웹캠으로 손을 인식하여 떨어지는 벚꽃 잎을 잡는 게임입니다.
- 손바닥을 화면에 보여주세요
- 손을 움직여 떨어지는 벚꽃 잎에 닿으면 잡힙니다
- Q 또는 ESC를 누르면 종료합니다

Requirements:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import os
import urllib.request


# ──────────────────────────────────────────────
# 상수 설정
# ──────────────────────────────────────────────
WINDOW_NAME = "Cherry Blossom Catcher"
PETAL_COUNT_INITIAL = 5          # 초기 꽃잎 수
PETAL_SPAWN_INTERVAL = 1.5       # 새 꽃잎 생성 간격 (초)
PETAL_CATCH_RADIUS = 50          # 잡기 판정 반경 (px)
GAME_DURATION = 60               # 게임 시간 (초)

# 벚꽃 색상 팔레트 (BGR)
PETAL_COLORS = [
    (180, 105, 255),   # 핫핑크
    (147, 112, 219),   # 미디엄 퍼플
    (203, 192, 255),   # 연핑크
    (170, 140, 255),   # 라벤더 핑크
    (180,  80, 210),   # 진핑크
]

# UI 색상
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (  0,   0,   0)
COLOR_GOLD    = ( 30, 215, 255)
COLOR_RED     = (  0,  50, 220)
COLOR_GREEN   = ( 80, 200,  80)
COLOR_OVERLAY = (255, 230, 245)  # 화면 틴트


# ──────────────────────────────────────────────
# 벚꽃 잎 클래스
# ──────────────────────────────────────────────
class Petal:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.reset()

    def reset(self):
        self.x = random.randint(30, self.frame_w - 30)
        self.y = random.randint(-60, -10)
        self.speed_y = random.uniform(2.0, 5.0)
        self.speed_x = random.uniform(-1.2, 1.2)
        self.size = random.randint(14, 26)
        self.color = random.choice(PETAL_COLORS)
        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-3.0, 3.0)
        self.alpha = random.uniform(0.7, 1.0)
        self.caught = False
        self.catch_anim = 0       # 잡힘 애니메이션 프레임 수

    def update(self):
        if self.caught:
            self.catch_anim += 1
            return self.catch_anim > 20  # 20프레임 후 제거

        self.x += self.speed_x
        self.y += self.speed_y
        self.angle += self.rot_speed

        # 좌우 화면 밖으로 나가면 반대편에서 등장
        if self.x < -30:
            self.x = self.frame_w + 20
        elif self.x > self.frame_w + 30:
            self.x = -20

        # 화면 아래로 떨어지면 사라짐 (missed)
        if self.y > self.frame_h + 30:
            return True  # 제거 신호
        return False

    def draw(self, frame: np.ndarray):
        cx, cy = int(self.x), int(self.y)

        if self.caught:
            # 잡힘 애니메이션: 점점 커지며 사라짐
            radius = self.size + self.catch_anim * 2
            alpha_fade = max(0, 1.0 - self.catch_anim / 20)
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), radius, self.color, -1)
            cv2.addWeighted(overlay, alpha_fade * 0.8, frame, 1 - alpha_fade * 0.8, 0, frame)
            return

        # 꽃잎 모양 (회전된 타원 5개로 구성)
        overlay = frame.copy()
        for i in range(5):
            petal_angle = math.radians(self.angle + i * 72)
            px = cx + int(math.cos(petal_angle) * self.size * 0.6)
            py = cy + int(math.sin(petal_angle) * self.size * 0.6)
            axes = (max(2, self.size // 2), max(1, self.size // 4))
            rotation_deg = int(math.degrees(petal_angle)) % 180
            cv2.ellipse(overlay, (px, py), axes, rotation_deg, 0, 360, self.color, -1)

        # 중심 원
        center_color = tuple(min(255, c + 60) for c in self.color)
        cv2.circle(overlay, (cx, cy), max(2, self.size // 5), center_color, -1)

        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)


# ──────────────────────────────────────────────
# 손 인식 래퍼 (MediaPipe Tasks API)
# ──────────────────────────────────────────────
# 손 랜드마크 연결 (시각화용)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

class HandTracker:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"

    def __init__(self):
        if not os.path.exists(self.MODEL_PATH):
            print("손 인식 모델 다운로드 중... (최초 1회, 수초 소요)")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("다운로드 완료!")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.start_time = time.time()

    def get_hand_centers(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """각 손의 중심 좌표 목록 반환"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        centers = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                wrist   = hand_landmarks[0]
                mid_mcp = hand_landmarks[9]
                cx = int((wrist.x + mid_mcp.x) / 2 * w)
                cy = int((wrist.y + mid_mcp.y) / 2 * h)
                centers.append((cx, cy))

                # 랜드마크 시각화
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (150, 100, 200), 1)
                for px, py in pts:
                    cv2.circle(frame, (px, py), 3, (200, 150, 255), -1)

        return centers

    def release(self):
        self.landmarker.close()


# ──────────────────────────────────────────────
# UI 헬퍼
# ──────────────────────────────────────────────
def draw_text_with_shadow(frame, text, pos, font_scale=1.0, color=COLOR_WHITE,
                           thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = pos
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, COLOR_BLACK, thickness + 1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_hud(frame, score: int, missed: int, time_left: float, combo: int):
    h, w = frame.shape[:2]

    # 상단 반투명 바
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 55), (40, 20, 60), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    # 점수
    draw_text_with_shadow(frame, f"SCORE  {score:04d}", (15, 38),
                          font_scale=1.0, color=COLOR_GOLD, thickness=2)

    # 남은 시간
    timer_color = COLOR_RED if time_left < 10 else COLOR_WHITE
    draw_text_with_shadow(frame, f"TIME  {max(0, int(time_left)):02d}s",
                          (w // 2 - 70, 38), font_scale=1.0, color=timer_color, thickness=2)

    # 놓침
    draw_text_with_shadow(frame, f"MISS  {missed:02d}", (w - 180, 38),
                          font_scale=1.0, color=(100, 160, 255), thickness=2)

    # 콤보
    if combo >= 3:
        combo_text = f"{combo} COMBO!"
        combo_color = COLOR_GOLD if combo < 10 else (0, 100, 255)
        draw_text_with_shadow(frame, combo_text, (w // 2 - 70, h - 20),
                              font_scale=1.1, color=combo_color, thickness=2)


def draw_hand_cursor(frame, centers: list[tuple[int, int]]):
    for cx, cy in centers:
        # 손 중심 표시: 십자선 + 원
        cv2.circle(frame, (cx, cy), PETAL_CATCH_RADIUS, (200, 100, 255), 1)
        cv2.drawMarker(frame, (cx, cy), (255, 180, 255),
                       cv2.MARKER_CROSS, 20, 1)


def draw_game_over(frame, score: int, missed: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (30, 0, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    draw_text_with_shadow(frame, "GAME OVER", (w // 2 - 155, h // 2 - 80),
                          font_scale=2.2, color=(180, 80, 255), thickness=3)
    draw_text_with_shadow(frame, f"Score: {score}", (w // 2 - 110, h // 2),
                          font_scale=1.4, color=COLOR_GOLD, thickness=2)
    draw_text_with_shadow(frame, f"Missed: {missed}", (w // 2 - 110, h // 2 + 50),
                          font_scale=1.1, color=(100, 160, 255), thickness=2)
    draw_text_with_shadow(frame, "R: Restart   Q: Quit", (w // 2 - 175, h // 2 + 110),
                          font_scale=0.9, color=COLOR_WHITE, thickness=1)


def draw_intro(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 0, 40), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    draw_text_with_shadow(frame, "Cherry Blossom Catcher", (w // 2 - 230, h // 2 - 90),
                          font_scale=2.0, color=(200, 120, 255), thickness=3)
    lines = [
        "Show your palm to the camera",
        "Move your hand to catch the petals!",
        "",
        "SPACE: Start   Q: Quit",
    ]
    for i, line in enumerate(lines):
        draw_text_with_shadow(frame, line, (w // 2 - 230, h // 2 - 10 + i * 40),
                              font_scale=0.85, color=COLOR_WHITE, thickness=1)


# ──────────────────────────────────────────────
# 메인 게임 루프
# ──────────────────────────────────────────────
def run_game():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인하세요.")
        return

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    tracker = HandTracker()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    # 게임 상태
    STATE_INTRO    = "intro"
    STATE_PLAYING  = "playing"
    STATE_GAMEOVER = "gameover"

    state = STATE_INTRO
    score = 0
    missed = 0
    combo = 0
    petals: list[Petal] = []
    last_spawn = 0.0
    game_start_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # 부드러운 핑크 틴트 오버레이
        tint = np.full_like(frame, COLOR_OVERLAY)
        cv2.addWeighted(tint, 0.08, frame, 0.92, 0, frame)

        hand_centers = tracker.get_hand_centers(frame)
        now = time.time()

        # ── INTRO ──────────────────────────────
        if state == STATE_INTRO:
            draw_intro(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                state = STATE_PLAYING
                score = 0
                missed = 0
                combo = 0
                petals = [Petal(w, h) for _ in range(PETAL_COUNT_INITIAL)]
                last_spawn = now
                game_start_time = now
            elif key in (ord('q'), ord('Q'), 27):
                break

        # ── PLAYING ────────────────────────────
        elif state == STATE_PLAYING:
            time_left = GAME_DURATION - (now - game_start_time)

            if time_left <= 0:
                state = STATE_GAMEOVER
                continue

            # 새 꽃잎 생성
            if now - last_spawn >= PETAL_SPAWN_INTERVAL:
                petals.append(Petal(w, h))
                last_spawn = now

            # 꽃잎 업데이트 & 충돌 판정
            caught_this_frame = 0
            to_remove = []
            for petal in petals:
                if petal.caught:
                    if petal.update():
                        to_remove.append(petal)
                    continue

                # 손과 충돌 판정
                caught = False
                for cx, cy in hand_centers:
                    dist = math.hypot(petal.x - cx, petal.y - cy)
                    if dist < PETAL_CATCH_RADIUS + petal.size:
                        petal.caught = True
                        caught = True
                        caught_this_frame += 1
                        break

                if not caught:
                    if petal.update():   # 화면 밖으로 나감
                        missed += 1
                        combo = 0
                        to_remove.append(petal)

            # 점수 계산 (콤보 보너스)
            if caught_this_frame > 0:
                combo += caught_this_frame
                bonus = 1 + (combo // 5)
                score += caught_this_frame * 10 * bonus

            for p in to_remove:
                petals.remove(p)

            # 그리기
            for petal in petals:
                petal.draw(frame)
            draw_hand_cursor(frame, hand_centers)
            draw_hud(frame, score, missed, time_left, combo)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

        # ── GAME OVER ──────────────────────────
        elif state == STATE_GAMEOVER:
            for petal in petals:
                petal.draw(frame)
            draw_game_over(frame, score, missed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('r'), ord('R')):
                state = STATE_PLAYING
                score = 0
                missed = 0
                combo = 0
                petals = [Petal(w, h) for _ in range(PETAL_COUNT_INITIAL)]
                last_spawn = now
                game_start_time = now
            elif key in (ord('q'), ord('Q'), 27):
                break

        cv2.imshow(WINDOW_NAME, frame)

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game()
