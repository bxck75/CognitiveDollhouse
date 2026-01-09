#!/usr/bin/env python3
"""
VideoSpriteSheet (single-row spritesheet)

Extract frames from a video and concatenate them into ONE horizontal row.
"""

import cv2
from PIL import Image


class VideoSpriteSheet:
    def __init__(
        self,
        video_path: str,
        every: int = 1,
        max_frames: int = 0,
        scale: float = 1.0,
        bg_color=(0, 0, 0),
    ):
        self.video_path = video_path
        self.every = max(1, every)
        self.max_frames = max(0, max_frames)
        self.scale = scale
        self.bg_color = bg_color

        self.frames: list[Image.Image] = []

    def extract_frames(self) -> list[Image.Image]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.frames.clear()
        idx = 0
        kept = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % self.every == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)

                if self.scale != 1.0:
                    w, h = 1568 / 2, 2272 /2
   
                    img = img.resize(
                        (int(w * self.scale), int(h * self.scale)),
                        Image.BICUBIC,
                    )

                self.frames.append(img)
                kept += 1

                if self.max_frames > 0 and kept >= self.max_frames:
                    break

            idx += 1

        cap.release()

        if not self.frames:
            raise RuntimeError("No frames extracted")

        return self.frames

    def make_spritesheet(self) -> Image.Image:
        if not self.frames:
            raise RuntimeError("No frames loaded. Call extract_frames() first.")

        fw, fh = self.frames[0].size
        count = len(self.frames)

        sheet_w = fw * count
        sheet_h = fh

        sheet = Image.new("RGB", (sheet_w, sheet_h), self.bg_color)

        x = 0
        for frame in self.frames:
            sheet.paste(frame, (x, 0))
            x += fw

        return sheet

    def save_spritesheet(self, output_path: str) -> None:
        sheet = self.make_spritesheet()
        sheet.save(output_path)

    def clear(self):
        self.frames.clear()



# example usage (remove or adapt)
if __name__ == "__main__":
    vss = VideoSpriteSheet(
        "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/video_sprites/4b1cffbd-b35e-4ed0-b77f-a87e5ad9ec95.mp4",
        every=8,
        max_frames=32,
        scale=0.6,
        bg_color=(255, 255, 255),
    )

    vss.extract_frames()
    vss.save_spritesheet("spritesheet2.png")
