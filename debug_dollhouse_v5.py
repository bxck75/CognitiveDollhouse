"""
Dollhouse v3: Integrated per-room agents with LLM responses + Bottom Console Bar.
Press Q to query agent, T to enter prompt. Console bar at bottom for all text I/O.
"""

import pygame
import random
import json
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

from dollhouse_agent import AgentManager, AgentResponse, PERSONAS
from dollhouse_worldscheduler import WorldScheduler

pygame.init()

# ============================================================================
# CONFIG
# ============================================================================

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900  # Extra space for console
FPS = 60

GRID_COLS = 3
GRID_ROWS = 3
ROOM_WIDTH = WINDOW_WIDTH // GRID_COLS
ROOM_HEIGHT = (WINDOW_HEIGHT - 100) // GRID_ROWS  # Leave 100px for console

COLOR_WALL = (245, 243, 240)
COLOR_FLOOR = (200, 180, 160)
COLOR_OUTLINE = (180, 170, 160)
COLOR_TEXT = (60, 60, 60)

CONSOLE_HEIGHT = 100
CONSOLE_BG = (30, 30, 35)
CONSOLE_TEXT = (200, 200, 200)
CONSOLE_ACCENT = (100, 180, 255)
CONSOLE_INPUT_BG = (50, 50, 60)

BASE_MODEL_PATH = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors"
LLM_MODEL_PATH="models/DarkIdol_Llama_3_1_8B_Instruct_1_2_Uncensored_Q6_K.gguf"
TEMPLATE_PATH = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/rooms/backups/bg_room.png"

# ============================================================================
# CONSOLE BAR
# ============================================================================

class ConsoleBar:
    """Bottom console bar for info display and text input with scrollable history"""
    
    def __init__(self, width: int, height: int, y_pos: int):
        self.width = width
        self.height = height
        self.y_pos = y_pos
        self.rect = pygame.Rect(0, y_pos, width, height)
        
        self.is_input_active = False
        self.input_text = ""
        self.input_cursor = 0
        self.blink_counter = 0
        
        self.log_lines: List[str] = []
        self.max_visible_lines = 3
        self.scroll_offset = 0  # How many lines scrolled up
        
        self.font_small = pygame.font.Font(None, 16)
        self.font_input = pygame.font.Font(None, 18)
    
    def add_log(self, message: str):
        """Add message to console log"""
        self.log_lines.append(message)
        # Auto-scroll to bottom when new message added
        self.scroll_offset = max(0, len(self.log_lines) - self.max_visible_lines)
    
    def clear_log(self):
        """Clear console log"""
        self.log_lines = []
        self.scroll_offset = 0
    
    def start_input(self):
        """Start text input mode"""
        self.is_input_active = True
        self.input_text = ""
        self.input_cursor = 0
    
    def end_input(self) -> str:
        """End input mode and return text"""
        self.is_input_active = False
        result = self.input_text
        self.input_text = ""
        self.input_cursor = 0
        return result
    
    def handle_event(self, event: pygame.event.EventType) -> Optional[str]:
        
        """Handle keyboard input, return completed input if RETURN pressed"""
        if not self.is_input_active:
            # Handle scroll input even when not typing
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PAGEUP:
                    self.scroll_offset = max(0, self.scroll_offset - 3)
                    return None
                elif event.key == pygame.K_PAGEDOWN:
                    self.scroll_offset = min(
                        max(0, len(self.log_lines) - self.max_visible_lines),
                        self.scroll_offset + 3
                    )
                    return None
                elif event.key == pygame.K_HOME:
                    self.scroll_offset = 0
                    return None
                elif event.key == pygame.K_END:
                    self.scroll_offset = max(0, len(self.log_lines) - self.max_visible_lines)
                    return None
            return None
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return self.end_input()
            elif event.key == pygame.K_ESCAPE:
                self.end_input()
                return None
            elif event.key == pygame.K_BACKSPACE:
                if self.input_cursor > 0:
                    self.input_text = self.input_text[:self.input_cursor-1] + self.input_text[self.input_cursor:]
                    self.input_cursor -= 1
            elif event.key == pygame.K_DELETE:
                if self.input_cursor < len(self.input_text):
                    self.input_text = self.input_text[:self.input_cursor] + self.input_text[self.input_cursor+1:]
            elif event.key == pygame.K_LEFT:
                self.input_cursor = max(0, self.input_cursor - 1)
            elif event.key == pygame.K_RIGHT:
                self.input_cursor = min(len(self.input_text), self.input_cursor + 1)
            elif event.key == pygame.K_HOME:
                self.input_cursor = 0
            elif event.key == pygame.K_END:
                self.input_cursor = len(self.input_text)
            elif event.unicode.isprintable():
                self.input_text = self.input_text[:self.input_cursor] + event.unicode + self.input_text[self.input_cursor:]
                self.input_cursor += 1
        
        return None
    
    def update(self):
        """Update blinking cursor"""
        self.blink_counter = (self.blink_counter + 1) % 30
    
    def draw(self, surface: pygame.Surface):
        """Draw console bar"""
        # Background
        pygame.draw.rect(surface, CONSOLE_BG, self.rect)
        pygame.draw.line(surface, CONSOLE_ACCENT, (0, self.y_pos), (self.width, self.y_pos), 2)
        
        # Log lines (with scrolling)
        visible_lines = self.log_lines[self.scroll_offset:self.scroll_offset + self.max_visible_lines]
        for i, line in enumerate(visible_lines):
            text_surf = self.font_small.render(line, True, CONSOLE_TEXT)
            surface.blit(text_surf, (10, self.y_pos + 5 + i * 18))
        
        # Scroll indicator
        if len(self.log_lines) > self.max_visible_lines:
            scroll_text = self.font_small.render(
                f"[{self.scroll_offset + self.max_visible_lines}/{len(self.log_lines)}]",
                True,
                (100, 100, 150)
            )
            surface.blit(scroll_text, (self.width - 120, self.y_pos + 5))
        
        # Input area
        if self.is_input_active:
            input_rect = pygame.Rect(10, self.y_pos + 60, self.width - 20, 28)
            pygame.draw.rect(surface, CONSOLE_INPUT_BG, input_rect)
            pygame.draw.rect(surface, CONSOLE_ACCENT, input_rect, 1)
            
            # Prompt label
            prompt_text = self.font_input.render("> ", True, CONSOLE_ACCENT)
            surface.blit(prompt_text, (input_rect.x + 5, input_rect.y + 5))
            
            # Input text
            text_display = self.input_text
            input_text_surf = self.font_input.render(text_display, True, CONSOLE_TEXT)
            surface.blit(input_text_surf, (input_rect.x + 25, input_rect.y + 5))
            
            # Cursor
            if self.blink_counter < 15:
                cursor_x = input_rect.x + 25 + self.font_input.size(text_display[:self.input_cursor])[0]
                pygame.draw.line(surface, CONSOLE_ACCENT, (cursor_x, input_rect.y + 5), (cursor_x, input_rect.y + 23), 2)
            
            help_text = self.font_small.render("ENTER: submit | ESC: cancel", True, (150, 150, 150))
            surface.blit(help_text, (self.width - 300, self.y_pos + 65))
        else:
            help_text = self.font_small.render(
                "Press T to enter prompt | PgUp/PgDn to scroll | Home/End to jump",
                True,
                (100, 100, 100)
            )
            surface.blit(help_text, (10, self.y_pos + 65))
            
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
    agent_response: Optional[AgentResponse] = None

# ============================================================================
# IMAGE MANAGEMENT
# ============================================================================

def pil_to_pygame(pil_image: Image.Image) -> pygame.Surface:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = np.transpose(arr, (1, 0, 2))
    arr = np.ascontiguousarray(arr)
    surface = pygame.surfarray.make_surface(arr)
    return surface

def pygame_to_pil(pygame_surface: pygame.Surface) -> Image.Image:
    arr = pygame.surfarray.array3d(pygame_surface)
    arr = np.transpose(arr, (1, 0, 2))
    arr = np.ascontiguousarray(arr)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def pygame_mask_to_pil(pygame_surface: pygame.Surface) -> Image.Image:
    arr = pygame.surfarray.array3d(pygame_surface)
    gray = arr[..., 0]
    gray = np.transpose(gray)
    gray = np.ascontiguousarray(gray)
    return Image.fromarray(gray, mode="L")

def load_room_background(room_id: int, visual_dir: Path = Path("visuals/rooms")) -> Optional[pygame.Surface]:
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
        if self.state.fade_progress < 1.0 and self.state.fade_duration > 0:
            self.state.fade_progress += 1.0 / self.state.fade_duration
            self.state.fade_progress = min(1.0, self.state.fade_progress)
            if self.state.fade_progress >= 1.0:
                self.background_image_prev = None
    
    def start_fade(self, duration_frames: float = 60.0):
        self.background_image_prev = self.background_image.copy() if self.background_image else None
        self.state.fade_progress = 0.0
        self.state.fade_duration = duration_frames
    
    def apply_new_image(self, pil_image: Image.Image):
        pygame_surf = pil_to_pygame(pil_image)
        pygame_surf = pygame.transform.scale(pygame_surf, (ROOM_WIDTH, ROOM_HEIGHT))
        
        self.start_fade(duration_frames=60.0)
        self.background_image = pygame_surf
        
        pygame.image.save(pygame_surf, f"visuals/rooms/bg_room_{self.state.room_id}.png")
    
    def undo_to_backup(self):
        restored = restore_backup(self.state.room_id, 0)
        if restored:
            self.start_fade(duration_frames=60.0)
            self.background_image = restored
    
    def reset_to_empty(self):
        template_path = Path(TEMPLATE_PATH)
        
        if not template_path.exists():
            print(f"✗ Template not found: {template_path}")
            return
        
        try:
            empty_surf = pygame.image.load(str(template_path))
            empty_surf = pygame.transform.scale(empty_surf, (ROOM_WIDTH, ROOM_HEIGHT))
            
            self.start_fade(duration_frames=60.0)
            self.background_image = empty_surf
            
            pygame.image.save(empty_surf, f"visuals/rooms/bg_room_{self.state.room_id}.png")
            print(f"✓ Room {self.state.room_id} reset to empty template")
        except Exception as e:
            print(f"✗ Reset failed: {e}")
    
    def draw(self, surface: pygame.Surface):
        room_rect = pygame.Rect(
            self.state.room_x * ROOM_WIDTH,
            self.state.room_y * ROOM_HEIGHT,
            ROOM_WIDTH,
            ROOM_HEIGHT
        )
        
        if self.background_image:
            if self.state.fade_progress < 1.0 and self.background_image_prev:
                fade_surf = self.background_image_prev.copy()
                new_surf = self.background_image.copy()
                new_surf.set_alpha(int(255 * self.state.fade_progress))
                fade_surf.blit(new_surf, (0, 0))
                surface.blit(fade_surf, (room_rect.x, room_rect.y))
            else:
                surface.blit(self.background_image, (room_rect.x, room_rect.y))
        else:
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
        pygame.display.set_caption("Dollhouse v3 — with Agents & Console")
        self.clock = pygame.time.Clock()
        self.world = WorldState()
        self.running = True
        
        self.focused_room_id = 0
        self.is_drawing_mask = False
        self.current_mask: Optional[pygame.Surface] = None
        self.brush_size = 20
        self.interaction_prompt = "a appartment interior with vibrant colorful graffiti, clear floor, retro wallpaper, 8k resolution"
        self.room_negative_prompt = "character, humanoid, blurry, bad quality"
        self.character_sprite_negative_prompt = "character, humanoid, blurry, bad quality"
        self.is_inferencing = False
        
        self.harvester = None
        self.agent_manager = None
        self.world_scheduler = WorldScheduler(fps=FPS)
        self.agent_query_pending = False
        self.agent_query_text = "What's on your mind?"
        
        # Initialize image generation folder
        Path("visuals/characters").mkdir(parents=True, exist_ok=True)
        Path("visuals/rooms").mkdir(parents=True, exist_ok=True)
        Path("visuals/rooms/backups").mkdir(parents=True, exist_ok=True)


        # Console bar
        self.console = ConsoleBar(WINDOW_WIDTH, CONSOLE_HEIGHT, WINDOW_HEIGHT - CONSOLE_HEIGHT)
        self.console.add_log("Dollhouse v3 ready | , / . : switch room | T: prompt | Q: query agent")
        
        self.inference_thread: Optional[threading.Thread] = None
        
        self._init_harvester()
        self._init_agents()
    
    def _init_harvester(self):
        try:
            from pipeline_harvester_v4 import PipelineHarvester
            self.console.add_log("Initializing harvester with RealESRGAN...")
            self.harvester = PipelineHarvester(
                model_path=BASE_MODEL_PATH,
                device="cuda",
                dtype=torch.float16,
                enable_attention_slicing=True,
                enable_upscaling=True,
            )
            self.console.add_log("✓ Harvester ready")
        except Exception as e:
            self.console.add_log(f"✗ Harvester failed: {str(e)[:50]}")
            self.harvester = None
    
    def _init_agents(self):
        try:
            self.console.add_log("Initializing agent manager...")
            self.agent_manager = AgentManager(model_path=LLM_MODEL_PATH, verbose=False)
            self.console.add_log("✓ Agent manager ready")
        except Exception as e:
            self.console.add_log(f"✗ Agent manager failed: {str(e)[:50]}")
            self.agent_manager = None
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Console input handling
            if self.console.is_input_active:
                result = self.console.handle_event(event)
                if result is not None:
                    # Check if this was an agent query
                    if self.agent_query_pending:
                        self.agent_query_text = result
                        self.console.add_log(f"Querying agent: {result}")
                        self._do_agent_query_threaded()
                        self.agent_query_pending = False
                    else:
                        # Regular prompt
                        self.interaction_prompt = result
                        self.console.add_log(f"Prompt: {result}")
                continue
            
            # Regular controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_COMMA:
                    self.focused_room_id = (self.focused_room_id - 1) % 9
                    persona = PERSONAS[self.focused_room_id]
                    self.console.add_log(f"Room {self.focused_room_id}: {persona.name}")
                elif event.key == pygame.K_PERIOD:
                    self.focused_room_id = (self.focused_room_id + 1) % 9
                    persona = PERSONAS[self.focused_room_id]
                    self.console.add_log(f"Room {self.focused_room_id}: {persona.name}")
                elif event.key == pygame.K_m:
                    self.is_drawing_mask = not self.is_drawing_mask
                    if self.is_drawing_mask:
                        self.current_mask = pygame.Surface((ROOM_WIDTH, ROOM_HEIGHT))
                        self.current_mask.fill((0, 0, 0))
                        self.console.add_log("Mask drawing: ON")
                    else:
                        self.console.add_log("Mask drawing: OFF")
                elif event.key == pygame.K_c:
                    self.current_mask = None
                    self.is_drawing_mask = False
                    self.console.add_log("Mask cleared")
                elif event.key == pygame.K_UP:
                    self.brush_size = min(100, self.brush_size + 5)
                    self.console.add_log(f"Brush size: {self.brush_size}")
                elif event.key == pygame.K_DOWN:
                    self.brush_size = max(5, self.brush_size - 5)
                    self.console.add_log(f"Brush size: {self.brush_size}")
                elif event.key == pygame.K_z:
                    room = list(self.world.rooms.values())[self.focused_room_id]
                    room.undo_to_backup()
                    self.console.add_log("Undid to backup")
                elif event.key == pygame.K_r:
                    room = list(self.world.rooms.values())[self.focused_room_id]
                    room.reset_to_empty()
                    self.console.add_log("Reset room to empty")
                elif event.key == pygame.K_q:
                    if self.agent_manager and not self.is_inferencing:
                        self._start_agent_query()
                elif event.key == pygame.K_RETURN:
                    if not self.is_inferencing and self.harvester:
                        self._start_inpaint_thread()
                elif event.key == pygame.K_BACKSPACE:
                    if not self.is_inferencing and self.harvester:
                        self._start_img2img_thread()
                elif event.key == pygame.K_s:
                    self._save_state()
                elif event.key == pygame.K_t:
                    self.console.start_input()
            
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
    
    def _start_agent_query(self):
            """Prompt user for agent query text"""
            self.console.add_log("Enter query for agent (T to edit):")
            self.console.start_input()
            self.agent_query_pending = True  # Flag to handle input result
        
    def _do_agent_query_threaded(self):
        """Query the focused room's agent (runs in thread)"""
        self.is_inferencing = True
        self.console.add_log("Agent thinking...")
        
        self.inference_thread = threading.Thread(
            target=self._do_agent_query,
            daemon=True
        )
        self.inference_thread.start()
    
    def _do_agent_query(self):
        """Execute agent query (runs in thread)"""
        try:
            if not self.agent_manager:
                self.console.add_log("✗ Agent unavailable")
                return
            
            room_id = self.focused_room_id
            persona = PERSONAS[room_id]
            
            world_context = self.world_scheduler.get_agent_context(room_id)
            
            # Use the query text provided by user
            response = self.agent_manager.query_agent(
                room_id,
                self.agent_query_text,  # Use user input
                world_context=world_context
            )
            
            room = list(self.world.rooms.values())[room_id]
            room.state.agent_response = response
            
            self.console.add_log(f"{persona.name}: {response.chat_txt} {response.emoji}")
            
            # AGENT AUTONOMOUS GENERATION - CHECK img2img_prompt
            if response.img2img_prompt and self.harvester:
                self.console.add_log("Agent rendering vision...")
                img_path = Path("visuals/rooms") / f"bg_room_{room_id}.png"
                
                if img_path.exists():
                    img = self.harvester.load_image_from_path(img_path)
                    result = self.harvester.generate_i2i(
                        prompt=response.img2img_prompt,
                        image=img,
                        strength=0.5,
                        num_inference_steps=20,
                        upscale_first=False,  # Don't upscale on agent regen
                    )
                    room.apply_new_image(result)
                    self.console.add_log(f"✓ {persona.name}'s room updated")
            
            # AGENT INPAINT - CHECK inpaint_prompt
            if response.inpaint_prompt and self.harvester:
                self.console.add_log(f"Agent decorating: {response.inpaint_region}")
                img_path = Path("visuals/rooms") / f"bg_room_{room_id}.png"
                
                if img_path.exists():
                    img = self.harvester.load_image_from_path(img_path)
                    # Create a simple centered mask for inpaint region
                    mask = self._create_agent_mask(response.inpaint_region)
                    
                    if mask:
                        result = self.harvester.generate_inpaint(
                            prompt=response.inpaint_prompt,
                            image=img,
                            mask=mask,
                            strength=0.75,
                            num_inference_steps=20,
                            upscale_first=False,
                        )
                        room.apply_new_image(result)
                        self.console.add_log(f"✓ Inpainted: {response.inpaint_region}")
        
        except Exception as e:
            print(f"Agent query error: {e}")
            import traceback
            traceback.print_exc()
            self.console.add_log(f"✗ Agent error: {str(e)[:40]}")
        
        finally:
            self.is_inferencing = False

    def _create_agent_mask(self, region: str) -> Optional[pygame.Surface]:
        """Create a mask based on inpaint region description"""
        mask = pygame.Surface((ROOM_WIDTH, ROOM_HEIGHT))
        mask.fill((0, 0, 0))
        
        region_lower = region.lower()
        
        # Map region descriptions to mask areas
        if "top" in region_lower or "ceiling" in region_lower:
            pygame.draw.rect(mask, (255, 255, 255), (0, 0, ROOM_WIDTH, ROOM_HEIGHT // 3))
        elif "bottom" in region_lower or "floor" in region_lower:
            pygame.draw.rect(mask, (255, 255, 255), (0, ROOM_HEIGHT * 2 // 3, ROOM_WIDTH, ROOM_HEIGHT // 3))
        elif "left" in region_lower:
            pygame.draw.rect(mask, (255, 255, 255), (0, 0, ROOM_WIDTH // 3, ROOM_HEIGHT))
        elif "right" in region_lower:
            pygame.draw.rect(mask, (255, 255, 255), (ROOM_WIDTH * 2 // 3, 0, ROOM_WIDTH // 3, ROOM_HEIGHT))
        elif "center" in region_lower or "middle" in region_lower:
            pygame.draw.ellipse(mask, (255, 255, 255), 
                              (ROOM_WIDTH // 4, ROOM_HEIGHT // 4, ROOM_WIDTH // 2, ROOM_HEIGHT // 2))
        else:
            # Default: whole room
            mask.fill((255, 255, 255))
        
        return mask
    

    def _start_inpaint_thread(self):
        if self.current_mask is None:
            self.console.add_log("✗ No mask drawn")
            return
        
        self.is_inferencing = True
        self.console.add_log("Inpainting...")
        
        self.inference_thread = threading.Thread(
            target=self._do_inpaint,
            daemon=True
        )
        self.inference_thread.start()
    
    def _do_inpaint(self):
        try:
            if not self.harvester:
                self.console.add_log("✗ Harvester not ready")
                return
            
            room_id = self.focused_room_id
            img_path = Path("visuals/rooms") / f"bg_room_{room_id}.png"
            
            if not img_path.exists():
                self.console.add_log("✗ Room image not found")
                return
            
            img = self.harvester.load_image_from_path(img_path)
            mask_pil = pygame_mask_to_pil(self.current_mask)
            
            with torch.no_grad():
                result = self.harvester.generate_inpaint(
                    prompt=self.interaction_prompt,
                    negative_prompt=self.room_negative_prompt,
                    image=img,
                    mask=mask_pil,
                    strength=0.75,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    seed=42,
                    upscale_first=True,
                )
            
            room = list(self.world.rooms.values())[room_id]
            room.apply_new_image(result)
            
            self.console.add_log("✓ Inpaint done")
        
        except Exception as e:
            print(f"Inpaint error: {e}")
            self.console.add_log(f"✗ Inpaint failed: {str(e)[:40]}")
        
        finally:
            self.is_inferencing = False
            self.current_mask = None
    
    def _start_img2img_thread(self):
        self.is_inferencing = True
        self.console.add_log("Regenerating...")
        
        self.inference_thread = threading.Thread(
            target=self._do_img2img,
            daemon=True
        )
        self.inference_thread.start()
    
    def _do_img2img(self):
        """Regenerate room from CLEAN TEMPLATE not degraded version"""
        try:
            if not self.harvester:
                self.console.add_log("✗ Harvester not ready")
                return
            
            room_id = self.focused_room_id
            
            # ALWAYS start from clean template, not current degraded image
            template_path = Path(TEMPLATE_PATH)
            
            if not template_path.exists():
                self.console.add_log("✗ Template not found")
                return
            
            # Load clean template instead of current room image
            img = self.harvester.load_image_from_path(template_path)
            
            with torch.no_grad():
                result = self.harvester.generate_i2i(
                    prompt=self.interaction_prompt,
                    negative_prompt=self.room_negative_prompt,
                    image=img,
                    strength=0.65,  # Slightly higher for more creativity
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    seed=None,  # Random seed each time
                    upscale_first=True,  # DO upscale for quality
                )
            
            room = list(self.world.rooms.values())[room_id]
            room.apply_new_image(result)
            
            self.console.add_log("✓ Img2img done (from clean template)")
        
        except Exception as e:
            print(f"Img2img error: {e}")
            self.console.add_log(f"✗ Img2img failed: {str(e)[:40]}")
        
        finally:
            self.is_inferencing = False

    
    def _save_state(self):
        state = {
            "frame": self.world.frame,
            "focused_room": self.focused_room_id,
        }
        with open("world_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        if self.agent_manager:
            self.agent_manager.save_responses()
        
        self.console.add_log("✓ State saved")
    
    def update(self):
        self.world.update()
        self.world_scheduler.update()
        self.console.update()
    
    def draw(self):
        self.screen.fill(COLOR_WALL)
        self.world.draw(self.screen)
        
        room_rect = self._get_room_rect()
        pygame.draw.rect(self.screen, (255, 200, 0), room_rect, 4)
        
        if self.is_drawing_mask and self.current_mask:
            mask_surf = self.current_mask.copy()
            mask_surf.set_alpha(100)
            self.screen.blit(mask_surf, (room_rect.x, room_rect.y))
        
        mouse_pos = pygame.mouse.get_pos()
        if room_rect.collidepoint(mouse_pos):
            pygame.draw.line(self.screen, (255, 100, 100),
                           (mouse_pos[0] - 10, mouse_pos[1]),
                           (mouse_pos[0] + 10, mouse_pos[1]), 2)
            pygame.draw.line(self.screen, (255, 100, 100),
                           (mouse_pos[0], mouse_pos[1] - 10),
                           (mouse_pos[0], mouse_pos[1] + 10), 2)
            pygame.draw.circle(self.screen, (255, 100, 100), mouse_pos, self.brush_size, 1)
        
        # Draw console bar
        self.console.draw(self.screen)
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        if self.harvester:
            self.harvester.cleanup()
        if self.agent_manager:
            self.agent_manager.cleanup()
        
        pygame.quit()

if __name__ == "__main__":
    app = DollhouseApp()
    app.run()