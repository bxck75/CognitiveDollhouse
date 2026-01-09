"""
Dollhouse Frontend: 9-room grid with wandering characters.
Live inference + fade transitions + 2-version backup rotation.
Fixed integration with PipelineHarvester v2.
"""

import pygame
import random
import json
import gc
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
import shutil
import threading
import torch

pygame.init()

# ============================================================================
# CONFIG
# ============================================================================

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800
FPS = 60

GRID_COLS = 3
GRID_ROWS = 3
ROOM_WIDTH = WINDOW_WIDTH // GRID_COLS
ROOM_HEIGHT = WINDOW_HEIGHT // GRID_ROWS

COLOR_WALL = (245, 243, 240)
COLOR_FLOOR = (200, 180, 160)
COLOR_OUTLINE = (180, 170, 160)
COLOR_TEXT = (60, 60, 60)

# Pipeline config
BASE_MODEL_PATH = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors"

# ============================================================================
# ENUMS & DATA
# ============================================================================

class PoseState(Enum):
    IDLE = "idle"
    WALK_LEFT = "walk_left"
    WALK_RIGHT = "walk_right"

@dataclass
class CharacterState:
    char_id: int
    room_x: int
    room_y: int
    local_x: float
    local_y: float
    pose: PoseState
    animation_frame: int
    appearance_seed: str
    color_tint: Tuple[int, int, int] = (200, 200, 200)

@dataclass
class RoomState:
    room_id: int
    room_x: int
    room_y: int
    background_seed: str
    fade_progress: float = 0.0
    fade_duration: float = 0.0

# ============================================================================
# IMAGE MANAGEMENT
# ============================================================================

def save_room_background(surface: pygame.Surface, room_id: int, visual_dir: Path = Path("visuals/rooms")):
    """Save with 2-version backup rotation"""
    visual_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = visual_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = visual_dir / f"bg_room_{room_id}.png"
    
    # Rotate backups: if current exists, backup it
    if img_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"bg_room_{room_id}_v{timestamp}.png"
        shutil.copy2(str(img_path), str(backup_path))
        
        # Keep only 2 most recent backups
        backups = sorted(backup_dir.glob(f"bg_room_{room_id}_v*.png"), reverse=True)
        if len(backups) > 2:
            backups[-1].unlink()  # Delete oldest
    
    pygame.image.save(surface, str(img_path))
    print(f"✓ Saved room {room_id}")

def load_room_background(room_id: int, visual_dir: Path = Path("visuals/rooms")) -> Optional[pygame.Surface]:
    """Load current room background"""
    img_path = visual_dir / f"bg_room_{room_id}.png"
    if not img_path.exists():
        return None
    try:
        img = pygame.image.load(str(img_path))
        return pygame.transform.scale(img, (ROOM_WIDTH, ROOM_HEIGHT))
    except Exception as e:
        print(f"Load failed: {e}")
        return None

def restore_backup(room_id: int, backup_index: int = 0, visual_dir: Path = Path("visuals/rooms")) -> Optional[pygame.Surface]:
    """Restore from backup (0=most recent)"""
    backup_dir = visual_dir / "backups"
    backups = sorted(backup_dir.glob(f"bg_room_{room_id}_v*.png"), reverse=True)
    
    if not backups or backup_index >= len(backups):
        return None
    
    try:
        img = pygame.image.load(str(backups[backup_index]))
        return pygame.transform.scale(img, (ROOM_WIDTH, ROOM_HEIGHT))
    except Exception as e:
        print(f"Restore failed: {e}")
        return None

# ============================================================================
# CHARACTER
# ============================================================================

class Character:
    def __init__(self, state: CharacterState):
        self.state = state
        self.idle_timer = 0
        self.idle_duration = random.randint(60, 180)
        self.direction = random.choice([-1, 1])
    
    def update(self):
        self.idle_timer += 1
        if self.idle_timer > self.idle_duration:
            if random.random() > 0.5:
                self.direction = random.choice([-1, 1])
                self.state.pose = PoseState.WALK_LEFT if self.direction == -1 else PoseState.WALK_RIGHT
            else:
                self.state.pose = PoseState.IDLE
            self.idle_timer = 0
            self.idle_duration = random.randint(60, 180)
        
        if self.state.pose != PoseState.IDLE:
            self.state.local_x += self.direction * 1.5
            room_left = self.state.room_x * ROOM_WIDTH + 40
            room_right = (self.state.room_x + 1) * ROOM_WIDTH - 40
            self.state.local_x = max(room_left, min(room_right, self.state.local_x))
            self.state.animation_frame = (self.state.animation_frame + 1) % 8
        else:
            self.state.animation_frame = 0
    
    def draw(self, surface: pygame.Surface):
        room_top = self.state.room_y * ROOM_HEIGHT
        local_y_in_room = self.state.local_y - room_top
        z_depth = local_y_in_room / ROOM_HEIGHT
        scale = 0.4 + z_depth * 0.6
        
        body_width = int(28 * scale)
        body_height = int(50 * scale)
        head_radius = int(10 * scale)
        y_adjust = int((50 - body_height) / 2)
        
        draw_x = int(self.state.local_x)
        draw_y = int(self.state.local_y - body_height + y_adjust)
        
        pygame.draw.rect(surface, self.state.color_tint,
                        (draw_x - body_width//2, draw_y, body_width, body_height),
                        border_radius=3)
        pygame.draw.circle(surface, (220, 180, 140), (draw_x, draw_y - head_radius), head_radius)
        
        if scale > 0.5:
            pygame.draw.circle(surface, (40, 40, 40), (draw_x - 3, draw_y - head_radius + 2), 2)
            pygame.draw.circle(surface, (40, 40, 40), (draw_x + 3, draw_y - head_radius + 2), 2)

# ============================================================================
# ROOM
# ============================================================================

class Room:
    def __init__(self, state: RoomState):
        self.state = state
        self.background_image: Optional[pygame.Surface] = None
        self.background_image_prev: Optional[pygame.Surface] = None
        self._load_background()
    
    def _load_background(self):
        self.background_image = load_room_background(self.state.room_id)
    
    def update(self):
        """Update fade transition"""
        if self.state.fade_progress < 1.0 and self.state.fade_duration > 0:
            self.state.fade_progress += 1.0 / self.state.fade_duration
            self.state.fade_progress = min(1.0, self.state.fade_progress)
            if self.state.fade_progress >= 1.0:
                self.background_image_prev = None
    
    def start_fade(self, duration_frames: float = 60.0):
        """Start fade transition"""
        self.background_image_prev = self.background_image.copy() if self.background_image else None
        self.state.fade_progress = 0.0
        self.state.fade_duration = duration_frames
    
    def apply_new_image(self, surface: pygame.Surface):
        """Apply new image and start fade"""
        self.start_fade(duration_frames=60.0)
        self.background_image = pygame.transform.scale(surface, (ROOM_WIDTH, ROOM_HEIGHT))
        save_room_background(self.background_image, self.state.room_id)
    
    def undo_to_backup(self):
        """Restore most recent backup"""
        restored = restore_backup(self.state.room_id, 0)
        if restored:
            self.apply_new_image(restored)
    
    def draw(self, surface: pygame.Surface):
        room_rect = pygame.Rect(
            self.state.room_x * ROOM_WIDTH,
            self.state.room_y * ROOM_HEIGHT,
            ROOM_WIDTH,
            ROOM_HEIGHT
        )
        
        if self.background_image:
            # Fade blend
            if self.state.fade_progress < 1.0 and self.background_image_prev:
                fade_surf = self.background_image_prev.copy()
                new_surf = self.background_image.copy()
                new_surf.set_alpha(int(255 * self.state.fade_progress))
                fade_surf.blit(new_surf, (0, 0))
                surface.blit(fade_surf, (room_rect.x, room_rect.y))
            else:
                surface.blit(self.background_image, (room_rect.x, room_rect.y))
        else:
            # Fallback
            pygame.draw.rect(surface, COLOR_WALL, room_rect)
            floor_height = int(ROOM_HEIGHT * 0.6)
            pygame.draw.rect(surface, COLOR_FLOOR,
                           (room_rect.x, room_rect.y + ROOM_HEIGHT - floor_height,
                            ROOM_WIDTH, floor_height))
        
        pygame.draw.rect(surface, COLOR_OUTLINE, room_rect, 2)

# ============================================================================
# WORLD
# ============================================================================

class WorldState:
    def __init__(self):
        self.characters: Dict[int, Character] = {}
        self.rooms: Dict[Tuple[int, int], Room] = {}
        self.frame = 0
        self._init_world()
    
    def _init_world(self):
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                room_id = row * GRID_COLS + col
                room_state = RoomState(room_id, col, row, f"room_{room_id}")
                self.rooms[(col, row)] = Room(room_state)
                
                char_state = CharacterState(
                    room_id, col, row,
                    col * ROOM_WIDTH + ROOM_WIDTH // 2,
                    row * ROOM_HEIGHT + ROOM_HEIGHT * 0.65,
                    PoseState.IDLE, 0, f"char_{room_id}",
                    (min(255, 100 + room_id * 15), 
                     min(255, 140 + room_id * 5),
                     min(255, 160 + room_id * 3))
                )
                self.characters[room_id] = Character(char_state)
    
    def update(self):
        for char in self.characters.values():
            char.update()
        for room in self.rooms.values():
            room.update()
        self.frame += 1
    
    def draw(self, surface: pygame.Surface):
        for room in self.rooms.values():
            room.draw(surface)
        
        sorted_chars = sorted(self.characters.values(), key=lambda c: c.state.local_y)
        for char in sorted_chars:
            char.draw(surface)

# ============================================================================
# APP
# ============================================================================

class DollhouseApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Dollhouse — Live Proto")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.world = WorldState()
        self.running = True
        
        self.focused_room_id = 0
        self.is_drawing_mask = False
        self.current_mask: Optional[pygame.Surface] = None
        self.brush_size = 20
        self.interaction_prompt = "vibrant colorful graffiti"
        
        # Inference state
        self.is_inferencing = False
        self.inference_progress = ""
        self.inference_thread: Optional[threading.Thread] = None
        
        # Try to import harvester
        self.harvester = None
        self._init_harvester()
    
    def _init_harvester(self):
        """Initialize pipeline harvester"""
        try:
            from pipeline_harvester_v2 import PipelineHarvester
            print("Initializing harvester...")
            self.harvester = PipelineHarvester(
                model_path=BASE_MODEL_PATH,
                device="cuda",
                dtype="float16",
                enable_attention_slicing=True,
            )
            print("✓ Harvester ready")
        except Exception as e:
            print(f"✗ Harvester failed: {e}")
            self.harvester = None
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_COMMA:
                    self.focused_room_id = (self.focused_room_id - 1) % 9
                elif event.key == pygame.K_PERIOD:
                    self.focused_room_id = (self.focused_room_id + 1) % 9
                elif event.key == pygame.K_m:
                    self.is_drawing_mask = not self.is_drawing_mask
                    if self.is_drawing_mask:
                        self.current_mask = pygame.Surface((ROOM_WIDTH, ROOM_HEIGHT))
                        self.current_mask.fill((0, 0, 0))
                elif event.key == pygame.K_c:
                    self.current_mask = None
                    self.is_drawing_mask = False
                elif event.key == pygame.K_UP:
                    self.brush_size = min(100, self.brush_size + 5)
                elif event.key == pygame.K_DOWN:
                    self.brush_size = max(5, self.brush_size - 5)
                elif event.key == pygame.K_z:
                    room = list(self.world.rooms.values())[self.focused_room_id]
                    room.undo_to_backup()
                elif event.key == pygame.K_RETURN:
                    if not self.is_inferencing and self.harvester:
                        self._start_inpaint_thread()
                elif event.key == pygame.K_BACKSPACE:
                    if not self.is_inferencing and self.harvester:
                        self._start_img2img_thread()
                elif event.key == pygame.K_s:
                    self._save_state()
                elif event.key == pygame.K_t:
                    # Type prompt
                    print("Enter prompt (or press ENTER to skip):")
                    self.interaction_prompt = input("> ") or self.interaction_prompt
            
            elif event.type == pygame.MOUSEMOTION:
                if self.is_drawing_mask and self.current_mask:
                    self._draw_on_mask(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.is_drawing_mask:
                    self._draw_on_mask(event.pos)
    
    def _get_room_rect(self) -> pygame.Rect:
        room_x = self.focused_room_id % GRID_COLS
        room_y = self.focused_room_id // GRID_COLS
        return pygame.Rect(room_x * ROOM_WIDTH, room_y * ROOM_HEIGHT, ROOM_WIDTH, ROOM_HEIGHT)
    
    def _draw_on_mask(self, pos: Tuple[int, int]):
        room_rect = self._get_room_rect()
        if not room_rect.collidepoint(pos):
            return
        local_x = pos[0] - room_rect.x
        local_y = pos[1] - room_rect.y
        pygame.draw.circle(self.current_mask, (255, 255, 255), (int(local_x), int(local_y)), self.brush_size)
    
    def _pygame_surface_to_pil(self, pygame_surface: pygame.Surface) -> Image.Image:
        """Convert pygame surface to PIL Image"""
        arr = pygame.surfarray.array3d(pygame_surface)
        arr = np.transpose(arr, (1, 0, 2))  # Swap axes for PIL
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    
    def _pygame_mask_to_pil(self, pygame_surface: pygame.Surface) -> Image.Image:
        """Convert pygame mask to PIL grayscale Image"""
        arr = pygame.surfarray.array3d(pygame_surface)
        gray = arr[:, :, 0]  # Use first channel
        gray = np.transpose(gray)
        return Image.fromarray(gray, mode="L")
    
    def _start_inpaint_thread(self):
        """Start inpaint in background thread"""
        if self.current_mask is None:
            self.inference_progress = "No mask drawn!"
            return
        
        self.is_inferencing = True
        self.inference_progress = "Inpainting..."
        
        # Create thread
        self.inference_thread = threading.Thread(
            target=self._do_inpaint,
            daemon=True
        )
        self.inference_thread.start()
    
    def _do_inpaint(self):
        """Execute inpaint (runs in thread)"""
        try:
            if not self.harvester:
                self.inference_progress = "✗ Harvester not ready"
                return
            
            room_id = self.focused_room_id
            img_path = Path("visuals/rooms") / f"bg_room_{room_id}.png"
            
            if not img_path.exists():
                self.inference_progress = "✗ Room image not found"
                return
            
            # Load image and mask
            img = self.harvester.load_image_from_path(img_path)
            mask_pil = self._pygame_mask_to_pil(self.current_mask)
            
            self.inference_progress = "Running inpaint..."
            
            # Generate
            with torch.no_grad():
                result = self.harvester.generate_inpaint(
                    prompt=self.interaction_prompt,
                    image=img,
                    mask=mask_pil,
                    strength=0.75,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    seed=42
                )
            
            # Convert PIL to pygame surface
            result_array = np.array(result)
            result_surf = pygame.surfarray.make_surface(
                np.transpose(result_array, (1, 0, 2))
            )
            
            # Apply to room
            room = list(self.world.rooms.values())[room_id]
            room.apply_new_image(result_surf)
            
            self.inference_progress = "✓ Inpaint done"
        
        except Exception as e:
            print(f"Inpaint error: {e}")
            import traceback
            traceback.print_exc()
            self.inference_progress = f"✗ {str(e)[:40]}"
        
        finally:
            self.is_inferencing = False
            self.current_mask = None
    
    def _start_img2img_thread(self):
        """Start img2img in background thread"""
        self.is_inferencing = True
        self.inference_progress = "Regenerating..."
        
        self.inference_thread = threading.Thread(
            target=self._do_img2img,
            daemon=True
        )
        self.inference_thread.start()
    
    def _do_img2img(self):
        """Execute img2img (runs in thread)"""
        try:
            if not self.harvester:
                self.inference_progress = "✗ Harvester not ready"
                return
            
            room_id = self.focused_room_id
            img_path = Path("visuals/rooms") / f"bg_room_{room_id}.png"
            
            if not img_path.exists():
                self.inference_progress = "✗ Room image not found"
                return
            
            img = self.harvester.load_image_from_path(img_path)
            
            self.inference_progress = "Running img2img..."
            
            with torch.no_grad():
                result = self.harvester.generate_i2i(
                    prompt=self.interaction_prompt,
                    image=img,
                    strength=0.6,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    seed=42
                )
            
            # Convert PIL to pygame surface
            result_array = np.array(result)
            result_surf = pygame.surfarray.make_surface(
                np.transpose(result_array, (1, 0, 2))
            )
            
            room = list(self.world.rooms.values())[room_id]
            room.apply_new_image(result_surf)
            
            self.inference_progress = "✓ Img2img done"
        
        except Exception as e:
            print(f"Img2img error: {e}")
            import traceback
            traceback.print_exc()
            self.inference_progress = f"✗ {str(e)[:40]}"
        
        finally:
            self.is_inferencing = False
    
    def _save_state(self):
        state = {
            "frame": self.world.frame,
            "focused_room": self.focused_room_id,
        }
        with open("world_state.json", "w") as f:
            json.dump(state, f, indent=2)
        print("✓ State saved")
    
    def update(self):
        self.world.update()
    
    def draw(self):
        self.screen.fill(COLOR_WALL)
        self.world.draw(self.screen)
        
        # Focus border
        room_rect = self._get_room_rect()
        pygame.draw.rect(self.screen, (255, 200, 0), room_rect, 4)
        
        # Mask overlay
        if self.is_drawing_mask and self.current_mask:
            mask_surf = self.current_mask.copy()
            mask_surf.set_alpha(100)
            self.screen.blit(mask_surf, (room_rect.x, room_rect.y))
        
        # Cursor
        mouse_pos = pygame.mouse.get_pos()
        if room_rect.collidepoint(mouse_pos):
            pygame.draw.line(self.screen, (255, 100, 100),
                           (mouse_pos[0] - 10, mouse_pos[1]),
                           (mouse_pos[0] + 10, mouse_pos[1]), 2)
            pygame.draw.line(self.screen, (255, 100, 100),
                           (mouse_pos[0], mouse_pos[1] - 10),
                           (mouse_pos[0], mouse_pos[1] + 10), 2)
            pygame.draw.circle(self.screen, (255, 100, 100), mouse_pos, self.brush_size, 1)
        
        # Info
        info = [
            f"Room {self.focused_room_id} (< >) | Brush {self.brush_size}px (↑ ↓)",
            f"Draw: {'ON (M off)' if self.is_drawing_mask else 'OFF (M on) C:clear'}",
            f"Prompt: {self.interaction_prompt[:40]}... (T:edit)",
            f"ENTER:inpaint | BACKSPACE:img2img | Z:undo | S:save | ESC:quit",
        ]
        if self.is_inferencing or self.inference_progress:
            info.append(f"[{self.inference_progress}]")
        
        for i, line in enumerate(info):
            text = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(text, (10, 10 + i * 25))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        # Cleanup
        if self.harvester:
            self.harvester.cleanup()
        
        pygame.quit()

if __name__ == "__main__":
    app = DollhouseApp()
    app.run()
