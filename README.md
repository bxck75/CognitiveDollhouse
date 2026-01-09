# 🏠 Dollhouse v3

> An interactive AI-powered dollhouse with per-room agents, real-time image generation, and dynamic world simulation.

---

## 🎮 Controls

### 🔄 Navigation & Room Selection
| Key | Action |
|-----|--------|
| **`,`** | Previous room (cycle left) |
| **`.`** | Next room (cycle right) |
| **`ESC`** | Exit application |

### 🎨 Image Generation & Editing
| Key | Action |
|-----|--------|
| **`ENTER`** | Inpaint (requires mask drawn with current prompt) |
| **`BACKSPACE`** | Full room regeneration (img2img with current prompt) |
| **`T`** | Enter/edit text prompt |

### 🖌️ Mask Drawing
| Key | Action |
|-----|--------|
| **`M`** | Toggle mask drawing mode |
| **`C`** | Clear current mask |
| **`↑`** | Increase brush size (+5px) |
| **`↓`** | Decrease brush size (-5px) |
| **`LMB`** | Draw on mask (click & drag) |

### 🤖 Agent System
| Key | Action |
|-----|--------|
| **`Q`** | Query focused room's agent (LLM response + optional room regen) |

### 🏠 Room Management
| Key | Action |
|-----|--------|
| **`Z`** | Undo to last backup |
| **`R`** | Reset room to empty template |

### 💾 Console & Logging
| Key | Action |
|-----|--------|
| **`PAGE UP`** | Scroll console up 3 lines |
| **`PAGE DOWN`** | Scroll console down 3 lines |
| **`HOME`** | Jump to oldest message |
| **`END`** | Jump to newest message |

### 📦 Utility
| Key | Action |
|-----|--------|
| **`S`** | Save world state & agent responses |

---

## 🎯 Visual Indicators

```
🟨 Yellow Border    → Currently focused room
🎯 Crosshair+Circle → Brush preview (hover over room)
📊 Console Messages → Real-time operation feedback
📈 [x/y] Counter   → Console scroll position
```

---

## 📚 Quick Workflows

### 🚀 Regenerate a Room
```
1. Select room with , or .
2. Press T → enter prompt (e.g., "cyberpunk neon room")
3. Press BACKSPACE → watch it regenerate
4. Check console for completion message
```

### 🎨 Inpaint Specific Area
```
1. Select room with , or .
2. Press M → enable mask drawing mode
3. Draw white areas where you want changes
   • Use ↑/↓ to adjust brush size (5-100px)
   • LMB to draw/drag
4. Press T → enter modification prompt
5. Press ENTER → inpaint begins
6. Press C → clear mask when done
```

### 💬 Chat with Agents
```
1. Select room with , or .
2. Press Q → agent thinks...
3. Agent responds in console
4. Room may regenerate based on agent's mood
5. Check emoji & mood in console output
```

### ↩️ Undo & Reset
```
• Press Z → restore to last backup
• Press R → reset to empty template
• Check console for confirmation
```

---

## 🏗️ Architecture Overview

### World State
- **9-room grid** (3×3 layout)
- **Per-room background images** with fade transitions
- **Character sprites** with depth-based scaling and animations

### Agent System
- **9 unique personas** (Luna, Kai, Sage, Blaze, Echo, Sunny, Raven, Ember, Nova)
- **Shared LLM instance** for memory efficiency
- **Structured JSON responses** with mood, thoughts, and optional room modifications

### World Scheduler
- **Day/night cycle** with brightness & time-of-day changes
- **Dynamic agent states** (energy, hunger, happiness, creativity, alertness)
- **Contextual event triggers** (sunrise, afternoon slump, sleep time, etc.)
- **Automatic state progression** for realistic behavior

### Image Generation
- **Stable Diffusion pipeline** with aggressive memory management
- **Three modes**: Text-to-Image, Image-to-Image, Inpainting
- **Optional RealESRGAN upscaling** for higher quality output
- **Non-blocking generation** (runs in background thread)

---

## 📦 Dependencies

```
pygame              # Graphics & windowing
torch               # GPU acceleration
diffusers           # Stable Diffusion pipelines
llama-cpp-python    # Local LLM inference
Pillow              # Image processing
numpy               # Numerical operations
transformers
diffusers   
peft
accelerate      
safetensors         # Model loading
mediapipe==0.10.13  # Pose estimation
```

---

## 🔧 Configuration

Edit top-level constants in `debug_dollhouse_v5.py`:

```python
WINDOW_WIDTH = 1600        # Screen width
WINDOW_HEIGHT = 900        # Screen height (includes console)
FPS = 60                   # Target frame rate

GRID_COLS = 3              # Room grid columns
GRID_ROWS = 3              # Room grid rows

BASE_MODEL_PATH = "..."    # Stable Diffusion model
LLM_MODEL_PATH = "..."     # Llama model
TEMPLATE_PATH = "..."      # Empty room template
```

---

## 📂 File Structure

```
.
├── debug_dollhouse_v5.py          # Main application
├── dollhouse_agent.py             # Agent system & personas
├── dollhouse_worldscheduler.py    # Time/event system
├── pipeline_harvester_v4.py       # Image generation pipeline
├── visuals/
│   └── rooms/
│       ├── bg_room_0.png          # Room backgrounds
│       ├── bg_room_1.png
│       └── backups/               # Backup versions
├── agent_logs/                    # Agent response history
└── world_state.json               # Saved world state
```

---

## 💡 Tips & Tricks

### 🎯 Best Prompts
- **Specific**: "cozy reading nook with warm lighting" (better than "room")
- **Atmospheric**: "cyberpunk, neon, rain on windows"
- **Detailed**: "wooden desk, potted plants, bookshelf, morning light"

### ⚡ Performance
- Use **lower inference steps** (16-20) for faster generation
- **Upscaling first** before img2img for better detail preservation
- Mask drawing is lightweight; generation runs in background

### 🎨 Creative Workflow
1. Start with img2img regeneration to establish baseline aesthetic
2. Use inpainting to refine specific areas
3. Query agents to get mood-based suggestions
4. Let the world scheduler provide natural narrative progression

### 💾 Backups
- Auto-saves backups before each generation
- `Z` restores to most recent backup
- `R` resets to empty template (useful for starting fresh)

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place models in correct paths
# - Stable Diffusion model → BASE_MODEL_PATH
# - Llama model → LLM_MODEL_PATH
# - Empty template → TEMPLATE_PATH

# 3. Run the application
python debug_dollhouse_v5.py
```

---

## 🤝 Agent Personas

| ID | Name | Role | Vibe |
|---|---|---|---|
| 0 | **Luna** | Artist & Dreamer | Whimsical, introspective |
| 1 | **Kai** | Tech Enthusiast | Curious, energetic |
| 2 | **Sage** | Philosopher | Calm, wise |
| 3 | **Blaze** | Adventurer | Bold, charismatic |
| 4 | **Echo** | Mysterious Wanderer | Quiet, cryptic |
| 5 | **Sunny** | Joyful Friend | Optimistic, warm |
| 6 | **Raven** | Academic | Intellectual, dry humor |
| 7 | **Ember** | Rebellious Artist | Provocative, expressive |
| 8 | **Nova** | Celestial Dreamer | Ethereal, spiritual |

---

## 📊 Console Output Examples

```
✓ Harvester ready
✓ Agent manager ready
Room 0: Luna
Prompt: vibrant colorful graffiti
Agent thinking...
Luna: I love the chaos of color. 🌙
Rendering agent vision...
✓ Room vision applied
```

---

## ⚙️ Advanced Usage

### Extend Agent Personas
Edit `PERSONAS` dictionary in `dollhouse_agent.py`:

```python
PERSONAS[9] = AgentPersona(
    name="Custom",
    role="Your role",
    personality="Your personality traits",
    appearance="Physical description",
    quirks="Behavioral quirks",
    interests=["interest1", "interest2"],
)
```

### Modify World Events
Edit `_init_events()` in `dollhouse_worldscheduler.py` to add custom events that trigger at specific times or conditions.

### Adjust Generation Parameters
Modify `strength`, `guidance_scale`, `num_inference_steps` in the generation methods for different quality/speed tradeoffs.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of GPU memory | Reduce resolution, lower `num_inference_steps`, enable `attention_slicing` |
| Slow generation | Lower inference steps, disable upscaling, use float32 instead of float16 |
| Agent not responding | Check `LLM_MODEL_PATH`, verify model file exists |
| Images not saving | Check `visuals/rooms/` directory exists and is writable |
| Console not updating | Ensure `world_scheduler.update()` is called in main loop |

---

## 📝 License

This is a personal project for experimental AI-driven interactive fiction.

---

**Made with 🎨 and 🤖**